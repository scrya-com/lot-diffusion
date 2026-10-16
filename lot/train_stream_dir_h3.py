#!/usr/bin/env python3
"""Streaming DiR (``stream_dir_h3``) trained by DMD against the frozen H3. Experimental.

    LOT_H3_TRAIN=1 python3 lot/train_stream_dir_h3.py --init lot/runs/dir_paper_run1 --init-tag _1000 \\
        --max-latent-hw 32 32 --steps 500 --out lot/runs/stream_run1

The same objective as ``train_dir_h3.py`` (see its docstring for the DMD signs and weights), on a
window of ``--frames`` (5n+2) latent frames instead of a still:

* window 0: frame 0 is the cache item's still latent, fed as the FL2VA ``first`` condition (I2V);
  frames 1.. start at ``r = 0``. No clean video is needed: DMD only reads text and shapes, and the
  base H3 supplies the video distribution.
* with probability ``--rolled``, a self-rolled window instead (Self-Forcing style): the generator
  streams window 0 without grad, slides, and the update is on window 1, whose frame 0 (the old
  frame 5) is the condition, frame 1 is carried done, and the memory holds one coarse keyframe.

Generator update: draw ``r`` in ``0..R-1``, advance the active frames ``r`` times without grad, then
once with grad. The real model (base H3, every LoRA off) and the fake model (fake LoRA + per-frame
``StreamFakeHead`` rows) see the same ``first`` condition as the generator and NO memory: the base
has never seen memory rows, and DMD needs both scores on the same conditioning. The DMD gradient and
the fake regression are masked to the active frames (done frames are fixed context).

So the memory is wired and trained only through what helps a window look like an H3 continuation of
its first frame. A real model that sees longer context (several FL2VA one-frame conditions) is the
next step; see the CHEATSHEET notes.

Warm start (``--init``): a still ``train_dir_h3.py`` run, its generator LoRA + ``DirHead`` into the
``StreamHead`` (done row and ``mem_proj`` zero) and its fake LoRA + fake head into the fake model.

Writes only under ``--out`` (default ``lot/runs/train_stream_dir_h3``, gitignored): ``log.jsonl``;
at ``--save-every`` and the end ``stream_head{tag}.pt``, ``stream_fake_head{tag}.pt``,
``lora_gen{tag}.safetensors``, ``lora_fake{tag}.safetensors``, ``config.json`` (``stream: true``), plus
``*_ema`` generator copies with ``--ema``. Load with ``stream_dir_h3.load_stream_run``. Refuses to start
unless ``LOT_H3_TRAIN=1`` (``--check`` is allowed).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before CUDA init

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lot_paths import LOT_H3_STILLS_CACHE  # noqa: E402

import train_h3  # noqa: E402
from dir_h3 import DEFAULT_R, probe_temb_dim, use_networks  # noqa: E402
from stream_dir_h3 import (StreamFakeHead, StreamHead, advance, check_window, memory_rows, ready,  # noqa: E402
                           slide, start_state, stream_forward, window_forward)
from train_dir_h3 import Ema, clip, grad_sum, make_optimizer, parse_tau, zero_grads  # noqa: E402
from h3_musubi import sample_sigma  # noqa: E402

TAU_MIN, TAU_MAX = 0.02, 0.98


class StreamRun:
    """One generator / fake update on one window; shared by ``--check``, training and eval."""

    def __init__(self, dit, head, fake_head, gen_net, fake_net, args, device):
        self.dit, self.head, self.fake_head = dit, head, fake_head
        self.gen_net, self.fake_net = gen_net, fake_net
        self.args, self.device = args, device
        self.iterations = head.iterations
        self.loss_scale = 1.0

    def draw_tau(self) -> float:
        dist = self.args.tau_shift
        if isinstance(dist, str) and dist.startswith("beta:"):
            tau = torch.distributions.Beta(float(dist.split(":", 1)[1]), 1.0).sample((1,))
        else:
            tau = sample_sigma(dist, self.device)
        return float(tau.clamp(TAU_MIN, TAU_MAX))

    @torch.no_grad()
    def window(self, still, text, rng: random.Random, seed: int, *, rolled: bool | None = None):
        """A window state ready for the update: window 0 (the still as ``first``) or, with probability
        ``--rolled``, window 1 after a no-grad stream of window 0. Then ``r`` no-grad iterations."""
        frames, height, width = self.args.frames, still.shape[-2], still.shape[-1]
        generator = torch.Generator(self.device).manual_seed(seed)
        state = start_state((frames, height, width), self.iterations, first=still[:, :, :1], generator=generator,
                            device=self.device)
        if rolled is None:
            rolled = rng.random() < self.args.rolled
        with use_networks(self.gen_net, self.fake_net, gen=True, fake=False):
            if rolled:
                while not ready(state, self.iterations):
                    advance(self.dit, self.head, state, text, memory_time=self.args.memory_time, audio_seed=seed + 1)
                slide(state, self.iterations, pool=self.args.pool, capacity=self.args.capacity, generator=generator)
            r = rng.randrange(self.iterations)
            for _ in range(r):
                advance(self.dit, self.head, state, text, memory_time=self.args.memory_time, audio_seed=seed + 1)
        return state, r, rolled

    def first(self, state):
        return state.estimate[:, :, :1]

    def scores(self, x_tau, tau, state, r_frames, text, seed):
        """``(x0_real, x0_fake)`` at ``tau``; both see the window's ``first`` condition and no memory."""
        audio = lambda: torch.Generator(self.device).manual_seed(seed + 99)  # noqa: E731
        with use_networks(self.gen_net, self.fake_net, gen=False, fake=False):
            real = x_tau + tau * window_forward(self.dit, x_tau, 1.0 - tau, text, first=self.first(state),
                                                audio_generator=audio())
        with use_networks(self.gen_net, self.fake_net, gen=False, fake=True):
            fake = x_tau + tau * window_forward(self.dit, x_tau, 1.0 - tau, text, first=self.first(state),
                                                frame_group=r_frames, group_delta=self.fake_head.fake_r_embed,
                                                audio_generator=audio())
        return real, fake

    def step_forward(self, state, text, seed):
        memory = memory_rows(state.memory, state.time, self.args.memory_time)
        return stream_forward(self.dit, self.head, state.z, state.estimate, state.r, text,
                              first=self.first(state) if state.has_first else None, memory=memory,
                              audio_generator=torch.Generator(self.device).manual_seed(seed + 1))

    def generator_update(self, still, text, rng, *, backward=True, rolled=None):
        """One DMD generator step on one window. Returns ``(loss, sample, stats)``; ``sample`` feeds the
        first fake update. Same no-grad-first ordering as ``train_dir_h3.Run.generator_update``."""
        seed = rng.randrange(2**31 - 1)
        state, r, rolled = self.window(still, text, rng, seed, rolled=rolled)
        active = state.r < self.iterations
        mask = active.to(self.device).view(1, 1, -1, 1, 1)
        r_frames = state.r.clone()
        with torch.no_grad(), use_networks(self.gen_net, self.fake_net, gen=True, fake=False):
            x0_hat = torch.where(mask, self.step_forward(state, text, seed), state.estimate)
        tau = self.draw_tau()
        eps = torch.randn_like(x0_hat)
        x_tau = ((1.0 - tau) * x0_hat + tau * eps).detach()
        with torch.no_grad():
            x0_real, x0_fake = self.scores(x_tau, tau, state, r_frames, text, seed)
            gap = ((x0_hat - x0_real).abs() * mask).sum() / (mask.sum() * x0_hat[:, :, :1].numel())
            weight = 1.0 / tau if self.args.dmd_weight == "paper" else 1.0 / float(gap.clamp(min=1e-6))
            grad = (x0_fake - x0_real) * weight * mask
        with use_networks(self.gen_net, self.fake_net, gen=True, fake=False), torch.enable_grad():
            x0_g = self.step_forward(state, text, seed)
            active_g = x0_g[:, :, active.to(x0_g.device)]
            target = (x0_g.detach() - grad)[:, :, active.to(x0_g.device)]
            loss = 0.5 * F.mse_loss(active_g, target)
            if backward:
                (loss * self.loss_scale).backward()
        on = active.to(self.device)
        stats = {"loss": float(loss.detach()), "tau": tau, "w": float(weight), "r": r, "rolled": int(rolled),
                 "active": int(on.sum()), "fake_real": float((x0_fake - x0_real)[:, :, on].abs().mean()),
                 "hat_real": float(gap), "hat_rms": float(x0_hat[:, :, on].pow(2).mean().sqrt()),
                 "match": float((x0_g.detach() - x0_hat)[:, :, on].abs().max())}
        return loss, (state, r_frames, x0_hat.detach(), text), stats

    def fake_update(self, sample, *, backward=True):
        """Flow regression of the fake model on a detached generator window, active frames only."""
        state, r_frames, x0_hat, text = sample
        active = (r_frames < self.iterations).to(self.device)
        tau = self.draw_tau()
        eps = torch.randn_like(x0_hat)
        x_tau = (1.0 - tau) * x0_hat + tau * eps
        with use_networks(self.gen_net, self.fake_net, gen=False, fake=True), torch.enable_grad():
            out = window_forward(self.dit, x_tau, 1.0 - tau, text, first=state.estimate[:, :, :1],
                                 frame_group=r_frames, group_delta=self.fake_head.fake_r_embed)
            loss = F.mse_loss(out[:, :, active], (x0_hat - eps)[:, :, active])   # H3 head: x0 - eps
            if backward:
                (loss * self.loss_scale).backward()
        return loss, tau

    def fresh_sample(self, still, text, rng):
        seed = rng.randrange(2**31 - 1)
        state, _r, _rolled = self.window(still, text, rng, seed)
        r_frames = state.r.clone()
        mask = (state.r < self.iterations).to(self.device).view(1, 1, -1, 1, 1)
        with torch.no_grad(), use_networks(self.gen_net, self.fake_net, gen=True, fake=False):
            x0_hat = torch.where(mask, self.step_forward(state, text, seed), state.estimate)
        return state, r_frames, x0_hat, text


def check(run: StreamRun, groups: dict, items, rng, device) -> None:
    """Window 0 and a rolled window: one generator and one fake update each; grads per group and memory."""
    still, text = train_h3.load_item(rng.choice(items), rng, device)
    with torch.no_grad():   # perturb the fake model so the DMD gradient is non-zero (check only)
        run.fake_head.fake_r_embed.normal_(0.0, 0.05)
        for module in run.fake_net.unet_loras:
            module.lora_up.weight.normal_(0.0, 0.02)
    for rolled in (False, True):
        all_params = [p for g in groups.values() for p in g]
        zero_grads(all_params)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        loss, sample, stats = run.generator_update(still, text, rng, rolled=rolled)
        torch.cuda.synchronize()
        sums = {k: grad_sum(v) for k, v in groups.items()}
        ok = (sums["gen_lora"] > 0 and sums["stream_head"] > 0 and sums["fake_lora"] == 0 and sums["fake_head"] == 0
              and math.isfinite(stats["loss"]))
        print(f"LOT_H3 kind=stream_check_gen rolled={int(rolled)} r={stats['r']} active={stats['active']} "
              f"loss={stats['loss']:.4g} regen_maxdiff={stats['match']:.3g} "
              + " ".join(f"grad_{k}={v:.3g}" for k, v in sums.items())
              + f" seconds={time.perf_counter() - start:.1f} peak_mb={torch.cuda.max_memory_allocated() / 2**20:.0f} "
              f"ok={int(ok)}", flush=True)
        zero_grads(all_params)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        floss, _tau = run.fake_update(sample)
        torch.cuda.synchronize()
        sums = {k: grad_sum(v) for k, v in groups.items()}
        ok = (sums["fake_lora"] > 0 and sums["fake_head"] > 0 and sums["gen_lora"] == 0 and sums["stream_head"] == 0
              and math.isfinite(float(floss.detach())))
        print(f"LOT_H3 kind=stream_check_fake rolled={int(rolled)} loss={float(floss.detach()):.4g} "
              + " ".join(f"grad_{k}={v:.3g}" for k, v in sums.items())
              + f" seconds={time.perf_counter() - start:.1f} peak_mb={torch.cuda.max_memory_allocated() / 2**20:.0f} "
              f"ok={int(ok)}", flush=True)
        zero_grads(all_params)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache", type=Path, action="append", default=None,
                        help=f"H3 cache dir (text + still latents); repeatable (default {LOT_H3_STILLS_CACHE})")
    parser.add_argument("--init", type=Path, default=None, help="still train_dir_h3 run to warm-start from")
    parser.add_argument("--init-tag", default="", help="checkpoint suffix in --init, e.g. _1000")
    parser.add_argument("--frames", type=int, default=7, help="window latent frames (5n+2)")
    parser.add_argument("--iterations", type=int, default=DEFAULT_R)
    parser.add_argument("--rolled", type=float, default=0.5, help="probability of a self-rolled (window 1) update")
    parser.add_argument("--pool", type=int, default=4)
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--memory-time", choices=["real", "packed"], default="real")
    parser.add_argument("--max-latent-hw", type=int, nargs=2, default=[32, 32], metavar=("H", "W"),
                        help="spatial crop cap in latent pixels (multiples of 8); a window has --frames times the rows")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--fake-updates", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--fake-lr", type=float, default=1e-4)
    parser.add_argument("--tau-shift", type=parse_tau, default="beta:2.5")
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--dmd-weight", choices=["norm", "paper"], default="paper")
    parser.add_argument("--betas", type=float, nargs=2, default=[0.0, 0.95], metavar=("B1", "B2"))   # the paper
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--ema", type=float, default=0.9999)   # the paper
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank without --init (else the init run's)")
    parser.add_argument("--swap", type=int, default=4)
    parser.add_argument("--holdout", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "runs" / "train_stream_dir_h3")
    parser.add_argument("--check", action="store_true", help="dry pass: grads and memory, writes nothing")
    args = parser.parse_args()

    check_window(args.frames)
    train_h3.MAX_LATENT_HW = tuple(args.max_latent_hw)
    if not args.check and os.environ.get("LOT_H3_TRAIN") != "1":
        raise SystemExit("streaming DiR training is gated: set LOT_H3_TRAIN=1 (or --check for a dry pass)")
    from gpu_guard import refuse_if_busy

    refuse_if_busy("train_stream_dir_h3.py")
    from h3_musubi import load_dit, make_lora

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    train_items, held_items = [], []
    for cache in args.cache or [LOT_H3_STILLS_CACHE]:
        train_part, held_part = train_h3.list_items(cache, args.holdout)
        train_items += train_part
        held_items += held_part

    dit = load_dit(device=device, swap=args.swap, train=True)
    hidden, temb_dim = dit.config.hidden_size, probe_temb_dim(dit)
    rank = args.rank
    if args.init is not None:
        from safetensors import safe_open

        with safe_open(str(args.init / f"lora_gen{args.init_tag}.safetensors"), "pt") as handle:
            key = next(k for k in handle.keys() if k.endswith("lora_down.weight"))
            rank = int(handle.get_slice(key).get_shape()[0])
    gen_net = make_lora(dit, rank)
    fake_net = make_lora(dit, rank)
    head = StreamHead(hidden, temb_dim, args.iterations)
    fake_head = StreamFakeHead(temb_dim, args.iterations)
    if args.init is not None:
        for net, name in ((gen_net, "lora_gen"), (fake_net, "lora_fake")):
            info = net.load_weights(str(args.init / f"{name}{args.init_tag}.safetensors"))
            if info.missing_keys:
                raise RuntimeError(f"--init {name} is missing {len(info.missing_keys)} keys")
        head.load_dir_head(torch.load(args.init / f"dir_head{args.init_tag}.pt", map_location="cpu", weights_only=True))
        fake_head.load_dir_fake_head(torch.load(args.init / f"dir_fake_head{args.init_tag}.pt", map_location="cpu",
                                                weights_only=True))
        print(f"LOT_H3 kind=stream_init from={args.init} tag={args.init_tag} rank={rank}", flush=True)
    for net in (gen_net, fake_net):
        net.requires_grad_(True)
        net.to(device=device, dtype=torch.bfloat16)
    head.to(device)
    fake_head.to(device)

    gen_lora_groups, _ = gen_net.prepare_optimizer_params(args.lr)
    fake_lora_groups, _ = fake_net.prepare_optimizer_params(args.fake_lr)
    gen_opt, opt_name = make_optimizer(gen_lora_groups + [{"params": list(head.parameters()), "lr": args.lr}], args.betas)
    fake_opt, _ = make_optimizer(fake_lora_groups + [{"params": list(fake_head.parameters()), "lr": args.fake_lr}],
                                 args.betas)
    groups = {"gen_lora": [p for g in gen_lora_groups for p in g["params"]], "stream_head": list(head.parameters()),
              "fake_lora": [p for g in fake_lora_groups for p in g["params"]], "fake_head": list(fake_head.parameters())}
    gen_params = groups["gen_lora"] + groups["stream_head"]
    fake_params = groups["fake_lora"] + groups["fake_head"]
    ema = Ema(gen_params, args.ema) if args.ema > 0 else None
    print(f"LOT_H3 kind=stream_start items={len(train_items)} holdout={len(held_items)} frames={args.frames} "
          f"iterations={args.iterations} rolled={args.rolled} crop={args.max_latent_hw} optimizer={opt_name} "
          f"gen_trainable={sum(p.numel() for p in gen_params)} swap={args.swap} "
          f"allocated_mb={torch.cuda.memory_allocated() / 2**20:.0f}", flush=True)

    run = StreamRun(dit, head, fake_head, gen_net, fake_net, args, device)
    if args.check:
        check(run, groups, train_items, rng, device)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    log = open(args.out / "log.jsonl", "a")
    meta = {"stream": True, "iters": args.iterations, "frames": args.frames, "rank": rank, "temb_dim": temb_dim,
            "hidden": hidden, "holdout": args.holdout, "cache": [str(c) for c in (args.cache or [LOT_H3_STILLS_CACHE])],
            "init": None if args.init is None else str(args.init), "init_tag": args.init_tag, "rolled": args.rolled,
            "pool": args.pool, "capacity": args.capacity, "memory_time": args.memory_time,
            "crop": args.max_latent_hw, "tau_shift": str(args.tau_shift), "fake_updates": args.fake_updates,
            "accum": args.accum, "dmd_weight": args.dmd_weight, "betas": list(args.betas), "grad_clip": args.grad_clip,
            "ema": args.ema, "ema_warmup": args.ema > 0}
    (args.out / "config.json").write_text(json.dumps(meta, indent=1))

    def save(tag: str) -> None:
        def write(sfx, gen_only):
            torch.save(head.state_dict(), args.out / f"stream_head{sfx}.pt")
            gen_net.save_weights(str(args.out / f"lora_gen{sfx}.safetensors"), torch.bfloat16,
                                 {"lot_h3": "stream_dir", "kind": "gen", "rank": str(rank)})
            if not gen_only:
                torch.save(fake_head.state_dict(), args.out / f"stream_fake_head{sfx}.pt")
                fake_net.save_weights(str(args.out / f"lora_fake{sfx}.safetensors"), torch.bfloat16,
                                      {"lot_h3": "stream_dir", "kind": "fake", "rank": str(rank)})

        write(tag, False)
        if ema is not None:
            with ema.swapped():
                write(f"{tag}_ema", True)
        print(f"LOT_H3 kind=stream_saved tag={tag or 'final'} out={args.out}", flush=True)

    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        gen_opt.zero_grad(set_to_none=True)
        samples, all_stats = [], []
        for _ in range(args.accum):
            still, text = train_h3.load_item(rng.choice(train_items), rng, device)
            run.loss_scale = 1.0 / args.accum
            _loss, sample, stats = run.generator_update(still, text, rng)
            if not math.isfinite(stats["loss"]):
                raise RuntimeError(f"step {step}: generator loss {stats['loss']}")
            samples.append(sample)
            all_stats.append(stats)
        run.loss_scale = 1.0
        grad_norm = clip(gen_params, args.grad_clip)
        gen_opt.step()
        if ema is not None:
            ema.update()
        gen_opt.zero_grad(set_to_none=True)
        fake_losses = []
        for k in range(args.fake_updates):
            fake_opt.zero_grad(set_to_none=True)
            for sample in samples:
                run.loss_scale = 1.0 / args.accum
                if k > 0:
                    still, text = train_h3.load_item(rng.choice(train_items), rng, device)
                    sample = run.fresh_sample(still, text, rng)
                floss, _tau = run.fake_update(sample)
                fake_losses.append(float(floss.detach()))
            run.loss_scale = 1.0
            clip(fake_params, args.grad_clip)
            fake_opt.step()
            fake_opt.zero_grad(set_to_none=True)
        stats = {key: sum(s[key] for s in all_stats) / len(all_stats) for key in all_stats[0]}
        record = {"step": step, **stats, "grad_norm": float(grad_norm),
                  "fake_loss": sum(fake_losses) / max(len(fake_losses), 1),
                  "seconds": round(time.perf_counter() - started, 1)}
        log.write(json.dumps(record) + "\n")
        if step % 10 == 0 or step == args.steps:
            log.flush()
            print(f"LOT_H3 kind=stream step={step} loss={stats['loss']:.4g} fake_loss={record['fake_loss']:.4f} "
                  f"fake_real={stats['fake_real']:.4f} hat_real={stats['hat_real']:.4f} grad_norm={grad_norm:.3g} "
                  f"s_per_step={(time.perf_counter() - started) / step:.1f} "
                  f"peak_mb={torch.cuda.max_memory_allocated() / 2**20:.0f}", flush=True)
        if args.save_every and step % args.save_every == 0:
            save(f"_{step}")
    save("")
    print(f"LOT_H3 kind=stream_done steps={args.steps} seconds={time.perf_counter() - started:.0f} ok=1", flush=True)


if __name__ == "__main__":
    main()
