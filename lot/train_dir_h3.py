#!/usr/bin/env python3
"""Data-space iteration (DiR) for H3, trained by DMD. Experimental.

    LOT_H3_TRAIN=1 python3 lot/train_dir_h3.py [--steps 1000] [--swap 4] [--schedule bands,bands,dense,dense --pairs DIR]

Paper: arXiv 2610.12102 (see ``dir_h3.py``). The generator is H3 plus a rank-``--rank`` LoRA
(``gen``), a zero-initialised projection of the previous estimate, and an embedding of the
iteration ``r``; it maps noise to a sample in ``R = 4`` iterations from the same noise. Training
is DMD against the frozen base model:

* real model: the base H3 (every LoRA off), flow time ``tau``;
* fake model: the base plus a second LoRA (``fake``, zero-up so fake == real at step 0) and its
  own iteration embedding, trained to denoise the generator's own samples.

Generator update: draw ``r`` from ``0..R-1``, roll ``r`` iterations without grad, then one with
grad gives ``x0_hat``. The silent-audio noise is drawn once per sample and is identical at every
iteration (like ``z``). Draw ``tau`` and noise, ``x_tau = (1 - tau) x0_hat + tau eps``. Both models
predict the clean latent from ``x_tau`` (no grad). With ``w = 1 / mean|x0_hat - x0_real|`` (DMD2's
normaliser, detached):

    grad = (x0_fake - x0_real) * w          loss = 0.5 * mse(x0_hat, (x0_hat - grad).detach())

(``--dmd-weight paper`` uses the paper's w(tau) = 1 instead: ``grad = (x0_fake - x0_real) / tau``.
``--betas``, ``--grad-clip`` and ``--ema`` expose the paper's optimizer settings; see CHEATSHEET.)

Real-data term (``--priming-weight``, off by default), after Priming Models (Park, Kim, Du, Chen,
Han; arXiv 2610.20331): their post-training loss F = D(a, y) - D(a, sg[a']) between generated
samples ``a``, real samples ``y`` and stop-gradient generated references ``a'``, with D an
entropic-OT cost in a frozen feature space. Here the feature space is the latent itself, ``y`` is
the cache item's own clean latent (one real sample per prompt, as their text-to-image setup) and
``a'`` are ``--priming-refs`` fresh no-grad rollouts at the same prompt and ``r``. With one live
sample per prompt both transport plans are uniform, so the OT cost is the mean squared distance
and the gradient is closed-form:

    s2 = 0.5 * (mean (a - y)^2 + mean_k mean (a - a'_k)^2)      (their pair-RMS scale, detached)
    grad_prime = (mean_k a'_k - y) / s2                          (same units/1/N convention as grad)

It moves every sample by the same displacement (the generator's mean toward ``y``), which is why
it does not collapse diversity the way plain regression to ``y`` would. The references use their
own random generator, so turning it on leaves every other draw (prompt, z, r, tau, eps) unchanged.

Sign: the loss' gradient with respect to ``x0_hat`` is ``(x0_hat - target) = grad / N``, so
descent moves ``x0_hat`` along ``-grad = (x0_real - x0_fake) * w``: toward what the real model
thinks is clean and away from what the fake model (the generator's own distribution) thinks.

Paper fidelity (appendix A). ``--paper-init`` initialises the added parameters as the paper does
(``DirHead.paper_init``: Xavier ``x0_proj`` without bias, ``r_embed`` and ``fake_r_embed`` from
N(0, 0.02)) instead of zeros; with zeros the estimate pathway starts dead and grows at most ~lr per
Adam step, which is why r = 4 looked like r = 1. ``--r-schedule mod`` trains iteration
``r = step mod R`` (the paper) instead of a random r per sample. ``--head-lr`` (not in the paper)
gives the head its own learning rate. Every default is the paper's (appendix A, Table 4):
``--paper-init``, ``--r-schedule mod``, ``--betas 0 0.95``, ``--ema 0.9999``, ``--dmd-weight paper``,
``--tau-shift beta:2.5``, ``--fake-updates 4``, ``--grad-clip 0``, ``--sigmas 1``, lr 1e-4. Runs before
2026-10-16 (run 1, the Turbo arms A-C) used ``--zero-init --r-schedule random``.

Fake update (``--fake-updates``, the paper does 4 per generator update): flow regression of the
fake model on ``x_tau`` built from detached generator samples, conditioned on ``r``, in H3's
``x0 - eps`` head convention. The first one reuses the generator step's sample; the others
draw fresh rollouts.

Layouts: ``--schedule`` gives one Level-of-Token layout per iteration (``dense``, ``bands``,
``center``, ``uniform2``, ``uniform4``; one name repeats). Any non-dense entry needs ``--pairs``
(the frozen bank, as ``train_h3.py``); ``--train-adapter`` also trains the adapter heads.

Data: the text embeddings and latent shapes of H3 cache items (``train_h3.list_items`` /
``load_item``; no clean latent is used). Torch memory: the grad forward and the no-grad
forwards are kept apart (see ``generator_update``) because the block-swap offloader switches
modes between them.

Writes only under ``--out`` (default ``lot/runs/train_dir_h3``, gitignored): ``log.jsonl`` and,
at ``--save-every`` and the end, ``dir_head.pt``, ``dir_fake_head.pt``, ``lora_gen.safetensors``,
``lora_fake.safetensors``, ``adapter.pt`` (LoT) and ``config.json`` (plus ``*_ema`` copies of the generator files with
``--ema``); load with
``dir_h3.load_dir_run``. Refuses to start unless ``LOT_H3_TRAIN=1`` (``--check`` is allowed).
"""

from __future__ import annotations

import argparse
import contextlib
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
from dir_h3 import (DEFAULT_R, DirFakeHead, DirHead, fake_velocity, fake_x0, generator_step,  # noqa: E402
                    next_state, parse_layout_schedule, parse_sigmas, probe_temb_dim, real_x0, save_dir_run,
                    use_networks)
from h3_musubi import sample_sigma  # noqa: E402
from h3_splice import LotSplice  # noqa: E402

TAU_MIN, TAU_MAX = 0.02, 0.98


# ------------------------------------------------------------------------------- pieces

class Run:
    """Everything one update needs, so ``--check``, training and eval share the same code."""

    def __init__(self, dit, head, fake_head, gen_net, fake_net, adapter, args, device):
        self.dit, self.head, self.fake_head = dit, head, fake_head
        self.gen_net, self.fake_net, self.adapter = gen_net, fake_net, adapter
        self.args, self.device = args, device
        self.iterations = args.iterations
        self.sigmas = parse_sigmas(args.sigmas, args.iterations)
        self.loss_scale = 1.0       # 1 / --accum while accumulating; backward() scales by it
        self._schedules: dict = {}
        # Priming references draw from their own stream so the global torch RNG (z, tau, eps) is
        # the same with and without --priming-weight: a paired A/B against a run without it.
        self.prime_rng = torch.Generator().manual_seed(args.seed + 7919)

    def schedule(self, z: torch.Tensor) -> list:
        """Layouts for ``z``'s token grid (cached per shape)."""
        key = tuple(z.shape[2:])
        if key not in self._schedules:
            t, h, w = key
            self._schedules[key] = parse_layout_schedule(self.args.schedule, self.iterations, h // 2, w // 2, t)
        return self._schedules[key]

    def splice(self, layout):
        return None if layout is None else LotSplice(self.adapter, layout, y_space=True)

    def audio(self, seed):
        return torch.Generator(self.device).manual_seed(seed)

    def one_step(self, state, estimate, r, text, schedule, audio_seed):
        """One iteration from the iteration state (``z`` itself with the default all-ones sigmas).
        The silent-audio noise is part of ``z``'s conditioning, so every call of one sample (all
        rollout iterations and the grad re-forward) gets a FRESH generator with the SAME
        ``audio_seed``: identical audio noise at every iteration."""
        audio = torch.Generator(self.device).manual_seed(audio_seed)
        return generator_step(self.dit, self.head, state, estimate, r, text, splice=self.splice(schedule[r]),
                              sigma=self.sigmas[r], audio_generator=audio)

    @torch.no_grad()
    def rollout(self, z, r, text, schedule, audio_seed) -> tuple[torch.Tensor, torch.Tensor]:
        """``(state x_r, x0_hat[r])`` (``(z, zeros)`` for r = 0), generator on, fake off, no grad."""
        estimate, state = torch.zeros_like(z), z
        with use_networks(self.gen_net, self.fake_net, gen=True, fake=False):
            for k in range(r):
                estimate = self.one_step(state, estimate, k, text, schedule, audio_seed)
                state = next_state(state, estimate, self.sigmas[k], self.sigmas[k + 1])
        return state, estimate

    def draw_tau(self) -> float:
        dist = self.args.tau_shift
        if isinstance(dist, str) and dist.startswith("beta:"):
            # The paper's importance sampling (Table 4, Algorithm 1): tau ~ Beta(a, 1), density a tau^(a-1).
            a = float(dist.split(":", 1)[1])
            tau = torch.distributions.Beta(a, 1.0).sample((1,))
        else:
            tau = sample_sigma(dist, self.device)
        return float(tau.clamp(TAU_MIN, TAU_MAX))

    # -------------------------------------------------------------------- generator update
    @torch.no_grad()
    def priming_grad(self, x0_hat, real, r, text, schedule):
        """Priming's latent-space real-data gradient for ``x0_hat`` (see the module docstring)."""
        refs = []
        for _ in range(self.args.priming_refs):
            z_ref = torch.randn(x0_hat.shape, generator=self.prime_rng).to(x0_hat.device, x0_hat.dtype)
            seed = int(torch.randint(0, 2**31 - 1, (1,), generator=self.prime_rng))
            state, prev = self.rollout(z_ref, r, text, schedule, seed)
            with use_networks(self.gen_net, self.fake_net, gen=True, fake=False):
                refs.append(self.one_step(state, prev, r, text, schedule, seed))
        cross = (x0_hat - real).pow(2).mean()
        self_term = sum((x0_hat - ref).pow(2).mean() for ref in refs) / len(refs)
        s2 = (0.5 * (cross + self_term)).clamp(min=1e-6)
        grad = (sum(refs) / len(refs) - real) / s2
        return grad, {"prime_cross": float(cross), "prime_self": float(self_term), "prime_s2": float(s2)}

    def generator_update(self, z, text, r, *, real=None, backward=True):
        """One DMD generator step. Returns ``(loss, x0_hat value, stats)``. ``real`` is the item's
        clean latent, used only by ``--priming-weight``.

        Order matters with block swap: the offloader is in forward-only mode under no_grad and
        in training mode for the grad forward, and switching modes between a grad forward and
        its backward would reset the block placement the backward hooks rely on. So every
        no-grad forward (rollout, ``x0_hat`` value, real, fake) runs first, then the grad
        forward (same audio noise, so the same ``x0_hat``) runs last and is followed straight
        by its backward. The extra forward costs about one DiT pass.
        """
        schedule = self.schedule(z)
        seed = int(torch.randint(0, 2**31 - 1, (1,)))
        state, estimate = self.rollout(z, r, text, schedule, seed)
        with torch.no_grad(), use_networks(self.gen_net, self.fake_net, gen=True, fake=False):
            x0_hat = self.one_step(state, estimate, r, text, schedule, seed)
        tau = self.draw_tau()
        eps = torch.randn_like(z)
        x_tau = ((1.0 - tau) * x0_hat + tau * eps).detach()
        with torch.no_grad():
            with use_networks(self.gen_net, self.fake_net, gen=False, fake=False):
                x0_real = real_x0(self.dit, x_tau, tau, text, audio_generator=self.audio(seed + 99))
            with use_networks(self.gen_net, self.fake_net, gen=False, fake=True):
                x0_fake = fake_x0(self.dit, self.fake_head, x_tau, tau, r, text, audio_generator=self.audio(seed + 99))
            gap = (x0_hat - x0_real).abs().mean().clamp(min=1e-6)
            if self.args.dmd_weight == "paper":
                # Paper eq. 10 / Algorithm 1: L = E w(tau) sg[v_real - v_fake]^T x0_hat, so
                # dL/dx0_hat = w(tau) (v_real - v_fake), with the paper's v = eps - x0 (target eps - x0_hat).
                # At a fixed x_tau = (1-tau) x0 + tau eps, x0 = x_tau - tau v, hence
                # v_real - v_fake = (x0_fake - x0_real) / tau. H3's head predicts x0 - eps = -v, which
                # is why fake_x0/real_x0 are x_tau + tau * out (the same x0 as x_tau - tau v).
                # The paper's final setting w(tau) = 1 ("velocity difference applied directly to x0_hat"):
                #     grad = (x0_fake - x0_real) / tau    (units of x0; descent moves x0_hat along -grad)
                # The surrogate below has dloss/dx0_hat = grad / N, the same 1/N scale as the paper's
                # mean(x_hat * direction). tau >= 0.02 here (TAU_MIN), so 1/tau <= 50.
                weight = 1.0 / tau
            else:
                weight = 1.0 / gap          # DMD2's normaliser (detached), the default
            grad = (x0_fake - x0_real) * weight
            grad_prime, prime_stats = None, {}
            if self.args.priming_weight > 0:
                if real is None:
                    raise ValueError("--priming-weight needs the item's real latent")
                grad_prime, prime_stats = self.priming_grad(x0_hat, real, r, text, schedule)
        with use_networks(self.gen_net, self.fake_net, gen=True, fake=False):
            with torch.enable_grad():
                x0_hat_g = self.one_step(state, estimate, r, text, schedule, seed)
                dmd = 0.5 * F.mse_loss(x0_hat_g, (x0_hat_g.detach() - grad))   # d/dx0_hat_g = grad / N
                loss = dmd
                ot = prime = None
                if grad_prime is not None:
                    prime = 0.5 * F.mse_loss(x0_hat_g, (x0_hat_g.detach() - grad_prime))   # d/dx0_hat_g = grad_prime / N
                    loss = loss + self.args.priming_weight * prime
                if self.args.ot_weight > 0:
                    ot = F.mse_loss(x0_hat_g, z)   # eq. 14: |G(x0_hat[r], z, r) - z|^2 (x - z = the head output)
                    loss = loss + self.args.ot_weight * ot
                if backward:
                    (loss * self.loss_scale).backward()
        stats = {"loss": float(loss.detach()), "dmd": float(dmd.detach()), "tau": tau, "w": float(weight),
                 "fake_real": float((x0_fake - x0_real).abs().mean()), "hat_real": float(gap),
                 "ot": None if ot is None else float(ot.detach()),
                 "grad_rms": float(grad.pow(2).mean().sqrt()),
                 "prime_rms": None if grad_prime is None else float(grad_prime.pow(2).mean().sqrt()),
                 **prime_stats,
                 "hat_rms": float(x0_hat.pow(2).mean().sqrt()),
                 "match": float((x0_hat_g.detach() - x0_hat).abs().max())}
        return loss, x0_hat.detach(), stats

    # ------------------------------------------------------------------------ fake update
    def fake_update(self, z, text, r, x0_hat=None, *, backward=True):
        """One fake-model flow-regression step on a detached generator sample. Returns ``(loss, tau)``."""
        if x0_hat is None:
            schedule = self.schedule(z)
            seed = int(torch.randint(0, 2**31 - 1, (1,)))
            with torch.no_grad(), use_networks(self.gen_net, self.fake_net, gen=True, fake=False):
                state, prev = self.rollout(z, r, text, schedule, seed)
                x0_hat = self.one_step(state, prev, r, text, schedule, seed)
        tau = self.draw_tau()
        eps = torch.randn_like(z)
        x_tau = (1.0 - tau) * x0_hat + tau * eps
        with use_networks(self.gen_net, self.fake_net, gen=False, fake=True), torch.enable_grad():
            out = fake_velocity(self.dit, self.fake_head, x_tau, tau, r, text)
            loss = F.mse_loss(out, x0_hat - eps)       # H3 head convention: x0 - eps
            if backward:
                (loss * self.loss_scale).backward()
        return loss, tau


class Ema:
    """fp32 exponential moving average of the generator's trainable parameters (gen LoRA, DirHead, adapter).

    Warm-up: update n (0-based) uses ``min(decay, (1 + n) / (10 + n))``. A plain 0.999 average started
    at step 0 still holds 0.999^600 = 55% of the untrained weights at step 600; with the warm-up the
    average covers roughly the last n/10 steps until n/10 reaches 1/(1 - decay) (n ~ 9000 at 0.999)."""

    def __init__(self, params, decay: float, updates: int = 0):
        self.params, self.decay, self.updates = list(params), float(decay), int(updates)
        self.shadow = [p.detach().float().clone() for p in self.params]

    def current_decay(self) -> float:
        return min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))

    @torch.no_grad()
    def update(self) -> None:
        decay = self.current_decay()
        for shadow, p in zip(self.shadow, self.params):
            shadow.mul_(decay).add_(p.detach().float(), alpha=1.0 - decay)
        self.updates += 1

    @contextlib.contextmanager
    def swapped(self):
        """Temporarily put the averaged values into the live parameters."""
        saved = [p.detach().clone() for p in self.params]
        with torch.no_grad():
            for shadow, p in zip(self.shadow, self.params):
                p.copy_(shadow.to(p.dtype))
        try:
            yield
        finally:
            with torch.no_grad():
                for old, p in zip(saved, self.params):
                    p.copy_(old)


def clip(params, max_norm: float) -> float:
    """Clip to ``max_norm`` (0 = off, norm only); returns the pre-clip norm."""
    return float(torch.nn.utils.clip_grad_norm_(params, max_norm if max_norm > 0 else float("inf")))


def grad_sum(params) -> float:
    return sum(float(p.grad.abs().sum()) for p in params if p.grad is not None)


def zero_grads(params) -> None:
    for p in params:
        p.grad = None


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp(min=1e-8))


# ------------------------------------------------------------------------------- check

def check(run: Run, groups: dict, items, rng, device) -> None:
    """One generator and one fake update (forward + backward) per distinct schedule entry.

    Prints grad sums per parameter group: the generator step must touch only gen LoRA / DirHead /
    adapter, the fake step only fake LoRA / fake head. Nothing is stepped or written.
    """
    x0, text = train_h3.load_item(rng.choice(items), rng, device)
    z = torch.randn_like(x0)
    # Fake == real at init (fake LoRA up and fake r-embed are zero), at a mid tau.
    saved_fake_r = run.fake_head.fake_r_embed.detach().clone()      # --paper-init: compare the LoRA alone
    with torch.no_grad():
        run.fake_head.fake_r_embed.zero_()
        tau = 0.5
        x_tau = (1.0 - tau) * torch.zeros_like(z) + tau * z
        # The silent-audio rows are random noise per call: reseed so only the LoRA/embedding differs.
        with use_networks(run.gen_net, run.fake_net, gen=False, fake=False):
            torch.manual_seed(123)
            real = real_x0(run.dit, x_tau, tau, text)
            torch.manual_seed(123)
            real2 = real_x0(run.dit, x_tau, tau, text)
        with use_networks(run.gen_net, run.fake_net, gen=False, fake=True):
            torch.manual_seed(123)
            fake = fake_x0(run.dit, run.fake_head, x_tau, tau, 0, text)
    with torch.no_grad():
        run.fake_head.fake_r_embed.copy_(saved_fake_r)
    init_gap, floor = rel(fake, real), rel(real2, real)
    print(f"LOT_H3 kind=dir_check_init fake_vs_real_rel={init_gap:.3g} run_to_run={floor:.3g} shape={tuple(z.shape)} "
          f"ok={int(init_gap < 1e-2)}", flush=True)

    # At init fake == real, so the DMD gradient is exactly zero. Perturb the fake model (check only,
    # nothing is saved) so the generator step has a signal to backpropagate.
    with torch.no_grad():
        run.fake_head.fake_r_embed.normal_(0.0, 0.05)
        for module in run.fake_net.unet_loras:
            module.lora_up.weight.normal_(0.0, 0.02)
    schedule = run.schedule(z)
    seen = set()
    for r, layout in enumerate(schedule):
        name = "dense" if layout is None else f"lot{layout.count}of{layout.dense_count}"
        if name in seen:
            continue
        seen.add(name)
        all_params = [p for g in groups.values() for p in g]
        zero_grads(all_params)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss, x0_hat, stats = run.generator_update(z, text, r, real=x0)
        torch.cuda.synchronize()
        gen_seconds = time.perf_counter() - start
        gen_peak = torch.cuda.max_memory_allocated() / 2**20
        sums = {k: grad_sum(v) for k, v in groups.items()}
        gen_ok = (sums["gen_lora"] > 0 and sums["dir_head"] > 0 and sums["fake_lora"] == 0 and sums["fake_head"] == 0
                  and (sums["adapter"] > 0 or not run.args.train_adapter or layout is None)
                  and math.isfinite(stats["loss"]))
        print(f"LOT_H3 kind=dir_check_gen r={r} layout={name} loss={stats['loss']:.4g} tau={stats['tau']:.3f} "
              f"w={stats['w']:.3g} fake_real={stats['fake_real']:.3g} hat_real={stats['hat_real']:.3g} "
              f"regen_maxdiff={stats['match']:.3g} grad_gen_lora={sums['gen_lora']:.3g} grad_dir_head={sums['dir_head']:.3g} "
              f"grad_adapter={sums['adapter']:.3g} grad_fake_lora={sums['fake_lora']:.3g} "
              f"grad_fake_head={sums['fake_head']:.3g} seconds={gen_seconds:.2f} peak_mb={gen_peak:.0f} ok={int(gen_ok)}",
              flush=True)
        if stats["prime_rms"] is not None:
            print(f"LOT_H3 kind=dir_check_priming r={r} grad_rms={stats['grad_rms']:.4g} prime_rms={stats['prime_rms']:.4g} "
                  f"weighted_ratio={run.args.priming_weight * stats['prime_rms'] / max(stats['grad_rms'], 1e-12):.3g} "
                  f"cross={stats['prime_cross']:.4g} self={stats['prime_self']:.4g} s2={stats['prime_s2']:.4g}", flush=True)
        zero_grads(all_params)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        floss, tau = run.fake_update(z, text, r, x0_hat)
        torch.cuda.synchronize()
        fake_seconds = time.perf_counter() - start
        sums = {k: grad_sum(v) for k, v in groups.items()}
        fake_ok = (sums["fake_lora"] > 0 and sums["fake_head"] > 0 and sums["gen_lora"] == 0 and sums["dir_head"] == 0
                   and sums["adapter"] == 0 and math.isfinite(float(floss.detach())))
        print(f"LOT_H3 kind=dir_check_fake r={r} layout={name} loss={float(floss.detach()):.4g} tau={tau:.3f} "
              f"grad_fake_lora={sums['fake_lora']:.3g} grad_fake_head={sums['fake_head']:.3g} "
              f"grad_gen_lora={sums['gen_lora']:.3g} grad_dir_head={sums['dir_head']:.3g} "
              f"grad_adapter={sums['adapter']:.3g} seconds={fake_seconds:.2f} "
              f"peak_mb={torch.cuda.max_memory_allocated() / 2**20:.0f} ok={int(fake_ok)}", flush=True)
        zero_grads(all_params)


# ------------------------------------------------------------------------------- main

def make_optimizer(groups, betas=(0.9, 0.999)):
    betas = tuple(betas)
    try:
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(groups, weight_decay=0.0, betas=betas), "adamw8bit"
    except ImportError:
        return torch.optim.AdamW(groups, weight_decay=0.0, betas=betas), "adamw"


def parse_tau(value: str):
    """``beta:A`` passes through (validated); anything else is ``train_h3.parse_shift``."""
    if value.startswith("beta:"):
        if float(value.split(":", 1)[1]) <= 0:
            raise argparse.ArgumentTypeError("beta:A needs A > 0")
        return value
    return train_h3.parse_shift(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache", type=Path, action="append", default=None,
                        help=f"H3 cache dir (text + shapes); repeatable (default {LOT_H3_STILLS_CACHE})")
    parser.add_argument("--schedule", default="dense",
                        help="layout per iteration, e.g. bands,bands,dense,dense (one name repeats); non-dense needs --pairs")
    parser.add_argument("--pairs", type=Path, default=None, help="Procrustes pair dir for the frozen bank (LoT schedules)")
    parser.add_argument("--iterations", type=int, default=DEFAULT_R, help="R, the number of iterations (paper: 4)")
    parser.add_argument("--init", type=Path, default=None,
                        help="warm-start the generator LoRA (lora_gen / lora.safetensors) and adapter from a run dir")
    parser.add_argument("--train-adapter", action="store_true", help="train the LoT adapter heads with the generator")
    parser.add_argument("--max-latent-hw", type=int, nargs=2, default=None, metavar=("H", "W"),
                        help="random spatial crop cap in latent pixels (multiples of 8), e.g. 48 48")
    parser.add_argument("--steps", type=int, default=1000, help="generator updates")
    parser.add_argument("--fake-updates", type=int, default=4,
                        help="fake-model updates per generator update (the paper: 4)")
    parser.add_argument("--lr", type=float, default=1e-4, help="generator learning rate (LoRA, DirHead, adapter)")
    parser.add_argument("--fake-lr", type=float, default=1e-4)
    parser.add_argument("--tau-shift", type=parse_tau, default="beta:2.5",
                        help="tau density: 'beta:A' (default beta:2.5, the paper), float shift (3 ~ Beta(2.5, 1)), "
                             "'sigmoid', or 'lognorm:S'; clamped to [0.02, 0.98]")
    parser.add_argument("--accum", type=int, default=1,
                        help="samples per generator and per fake update (gradient accumulation; the paper's batch is 256)")
    parser.add_argument("--dmd-weight", choices=["norm", "paper"], default="paper",
                        help="paper (default): w(tau) = 1 on the velocity difference, grad = (x0_fake - x0_real) / tau; "
                             "norm: grad = (x0_fake - x0_real) / mean|x0_hat - x0_real| (DMD2)")
    parser.add_argument("--betas", type=float, nargs=2, default=[0.0, 0.95], metavar=("B1", "B2"),
                        help="AdamW betas for both optimizers (default 0.0 0.95, the paper)")
    parser.add_argument("--grad-clip", type=float, default=0.0, help="gradient-norm clip, generator and fake (default 0 = off, the paper)")
    parser.add_argument("--ema", type=float, default=0.9999,
                        help="EMA decay of the generator's trainable params (default 0.9999, the paper; 0 = off, e.g. rank "
                             "128 on 24 GB), with warm-up min(D, (1+n)/(10+n)); saved as *_ema files")
    parser.add_argument("--ot-weight", type=float, default=0.0, help="paper eq. 14 regulariser weight (0 = off; paper: no gain)")
    parser.add_argument("--sigmas", default="1",
                        help="flow sigma per iteration: 1 (the paper: pinned), grid[:SHIFT] (H3's Euler grid, shift 12; any "
                             "init), turbo[:SHIFT] (shift 6, with a Turbo --init), or R comma values; see dir_h3")
    init = parser.add_mutually_exclusive_group()
    init.add_argument("--paper-init", dest="paper_init", action="store_true", default=True,
                      help="(default) the paper's init of the added params: Xavier x0_proj (no bias), r_embed and "
                           "fake_r_embed ~ N(0, 0.02). Ignored for files loaded by --init/--resume")
    init.add_argument("--zero-init", dest="paper_init", action="store_false",
                      help="zero-init the added params instead (step 0 == base; runs before 2026-10-16). The estimate "
                           "pathway then starts dead and grows ~lr per Adam step")
    parser.add_argument("--r-schedule", choices=["mod", "random"], default="mod",
                        help="iteration trained each step: step mod R for the whole step (default, the paper), or random per sample")
    parser.add_argument("--head-lr", type=float, default=None,
                        help="learning rate of DirHead (x0_proj, r_embed); default --lr. Not in the paper")
    parser.add_argument("--priming-weight", type=float, default=0.0,
                        help="Priming Models' real-data term in latent space (arXiv 2610.20331; 0 = off), see docstring")
    parser.add_argument("--priming-refs", type=int, default=1,
                        help="no-grad reference rollouts per sample for --priming-weight (each costs r + 1 forwards)")
    parser.add_argument("--rank", type=int, default=16, help="generator LoRA rank (and the fake's, unless --fake-rank)")
    parser.add_argument("--fake-rank", type=int, default=None, help="fake-model LoRA rank (default: --rank)")
    parser.add_argument("--swap", type=int, default=4, help="blocks streamed from CPU; >0 turns on checkpointing")
    parser.add_argument("--holdout", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=0, help="0 = save only at the end")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "runs" / "train_dir_h3")
    parser.add_argument("--resume-step", type=int, default=0,
                        help="continue the run in --out from its step-N save (lora_gen_N, lora_fake_N, dir_head_N, "
                             "dir_fake_head_N, *_N_ema, adapter_N); optimizer state is not saved and restarts")
    parser.add_argument("--check", action="store_true",
                        help="one generator + one fake update per distinct schedule entry, grads and memory printed, "
                             "no optimizer step, writes nothing; allowed without LOT_H3_TRAIN")
    args = parser.parse_args()

    train_h3.MAX_LATENT_HW = tuple(args.max_latent_hw) if args.max_latent_hw else None
    if not args.check and os.environ.get("LOT_H3_TRAIN") != "1":
        raise SystemExit("DiR training is gated: set LOT_H3_TRAIN=1 to train (or --check for a dry pass)")
    from gpu_guard import refuse_if_busy

    refuse_if_busy("train_dir_h3.py")
    from h3_musubi import load_dit, make_lora

    names = [n.strip() for n in args.schedule.split(",") if n.strip()]
    lot = any(n != "dense" for n in names)
    if lot and args.pairs is None:
        raise SystemExit("a non-dense --schedule needs --pairs (the frozen bank)")
    if not lot:
        train_h3.CROP_MULTIPLE = 2   # no LoT super-cells: keep the whole latent (H3's 768×1344 is 48×84)

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    train_items, held_items = [], []
    for cache in args.cache or [LOT_H3_STILLS_CACHE]:
        train_part, held_part = train_h3.list_items(cache, args.holdout)
        train_items += train_part
        held_items += held_part

    adapter = None
    if lot:
        from h3 import make_h3_adapter
        from procrustes_h3 import CHECKPOINT, fit_bank, load_pairs, pretrained_maps

        adapter = make_h3_adapter()
        adapter.init_from_pretrained(*pretrained_maps(CHECKPOINT))
        fit_bank(adapter, load_pairs(args.pairs))
        adapter.to(device)
        adapter.bank.requires_grad_(False)

    dit = load_dit(device=device, swap=args.swap, train=True)
    hidden, temb_dim = dit.config.hidden_size, probe_temb_dim(dit)
    gen_net = make_lora(dit, args.rank)          # applied first
    fake_net = make_lora(dit, args.fake_rank or args.rank)   # wraps the generator LoRA's forward; .enabled switches each off
    if args.init is not None:
        lora_file = next(f for f in (args.init / "lora_gen.safetensors", args.init / "lora.safetensors") if f.is_file())
        info = gen_net.load_weights(str(lora_file))
        if info.missing_keys:
            raise RuntimeError(f"--init LoRA is missing {len(info.missing_keys)} keys")
        if adapter is not None and ((args.init / "adapter.pt").is_file() or (args.init / "adapter.safetensors").is_file()):
            adapter.load_state_dict(train_h3.load_adapter_state(args.init))
        print(f"LOT_H3 kind=dir_init from={args.init}", flush=True)
    for net in (gen_net, fake_net):
        net.requires_grad_(True)
        net.to(device=device, dtype=torch.bfloat16)
    head = DirHead(hidden, temb_dim, args.iterations)
    fake_head = DirFakeHead(temb_dim, args.iterations)
    if args.paper_init:
        init_rng = torch.Generator().manual_seed(args.seed + 31)
        head.paper_init(init_rng)
        fake_head.paper_init(init_rng)
    head, fake_head = head.to(device), fake_head.to(device)
    if args.init is not None and (args.init / "dir_head.pt").is_file():
        head.load_state_dict(torch.load(args.init / "dir_head.pt", map_location=device, weights_only=True))

    gen_lora_groups, _ = gen_net.prepare_optimizer_params(args.lr)
    fake_lora_groups, _ = fake_net.prepare_optimizer_params(args.fake_lr)
    adapter_params = []
    if adapter is not None:
        adapter_params = [p for n, p in adapter.named_parameters() if not n.startswith("bank.")]
        for p in adapter_params:
            p.requires_grad_(args.train_adapter)
        if not args.train_adapter:
            adapter_params = []
    head_params = [p for p in head.parameters() if p.requires_grad]
    gen_groups = gen_lora_groups + [{"params": head_params, "lr": args.head_lr or args.lr}]
    if adapter_params:
        gen_groups.append({"params": adapter_params, "lr": args.lr})
    fake_groups = fake_lora_groups + [{"params": list(fake_head.parameters()), "lr": args.fake_lr}]
    gen_opt, opt_name = make_optimizer(gen_groups, args.betas)
    fake_opt, _ = make_optimizer(fake_groups, args.betas)
    param_groups = {
        "gen_lora": [p for g in gen_lora_groups for p in g["params"]],
        "dir_head": head_params,
        "adapter": adapter_params,
        "fake_lora": [p for g in fake_lora_groups for p in g["params"]],
        "fake_head": list(fake_head.parameters()),
    }
    gen_params = param_groups["gen_lora"] + param_groups["dir_head"] + adapter_params
    fake_params = param_groups["fake_lora"] + param_groups["fake_head"]
    if args.resume_step:
        # With the paper's betas (0, 0.95) Adam remembers ~20 steps, so a fresh optimizer costs little.
        def load_gen(tag):
            info = gen_net.load_weights(str(args.out / f"lora_gen{tag}.safetensors"))
            if info.missing_keys:
                raise RuntimeError(f"resume: lora_gen{tag} is missing {len(info.missing_keys)} keys")
            head.load_state_dict(torch.load(args.out / f"dir_head{tag}.pt", map_location=device, weights_only=True))
            if adapter is not None and (args.out / f"adapter{tag}.pt").is_file():
                adapter.load_state_dict(torch.load(args.out / f"adapter{tag}.pt", map_location=device, weights_only=True))

        tag = f"_{args.resume_step}"
        old_cfg = args.out / "config.json"
        warm_saved = old_cfg.is_file() and json.loads(old_cfg.read_text()).get("ema_warmup", False)
        if args.ema > 0 and warm_saved and (args.out / f"lora_gen{tag}_ema.safetensors").is_file():
            load_gen(f"{tag}_ema")                     # seed the EMA shadow with the saved average
            ema = Ema(gen_params, args.ema, updates=args.resume_step)
        else:
            ema = None
        load_gen(tag)
        if args.ema > 0 and ema is None:
            # No average, or one saved without warm-up (mostly untrained weights): restart it from the live
            # weights, with the warm-up counter at 0 so it tracks the recent steps.
            ema = Ema(gen_params, args.ema)
            print(f"LOT_H3 kind=dir_ema_reseed step={args.resume_step} from=live", flush=True)
        info = fake_net.load_weights(str(args.out / f"lora_fake{tag}.safetensors"))
        if info.missing_keys:
            raise RuntimeError(f"resume: lora_fake{tag} is missing {len(info.missing_keys)} keys")
        fake_head.load_state_dict(torch.load(args.out / f"dir_fake_head{tag}.pt", map_location=device, weights_only=True))
        print(f"LOT_H3 kind=dir_resume step={args.resume_step} from={args.out}", flush=True)
    else:
        ema = Ema(gen_params, args.ema) if args.ema > 0 else None
    print(f"LOT_H3 kind=dir_start items={len(train_items)} holdout={len(held_items)} iterations={args.iterations} "
          f"schedule={args.schedule} gen_lora_modules={len(gen_net.unet_loras)} fake_lora_modules={len(fake_net.unet_loras)} "
          f"gen_trainable={sum(p.numel() for p in gen_params)} fake_trainable={sum(p.numel() for p in fake_params)} "
          f"optimizer={opt_name} swap={args.swap} temb_dim={temb_dim} hidden={hidden} "
          f"fake_updates={args.fake_updates} accum={args.accum} tau_shift={args.tau_shift} dmd_weight={args.dmd_weight} "
          f"betas={tuple(args.betas)} grad_clip={args.grad_clip} ema={args.ema} priming={args.priming_weight}x{args.priming_refs} sigmas={args.sigmas} paper_init={int(args.paper_init)} r_schedule={args.r_schedule} head_lr={args.head_lr or args.lr} "
          f"allocated_mb={torch.cuda.memory_allocated() / 2**20:.0f}", flush=True)

    run = Run(dit, head, fake_head, gen_net, fake_net, adapter, args, device)
    if args.check:
        check(run, param_groups, train_items, rng, device)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    log = open(args.out / "log.jsonl", "a")
    meta = {"iters": args.iterations, "holdout": args.holdout,
            "cache": [str(c) for c in (args.cache or [LOT_H3_STILLS_CACHE])], "schedule": args.schedule, "rank": args.rank, "fake_rank": args.fake_rank or args.rank, "temb_dim": temb_dim,
            "hidden": hidden, "lot": lot, "pairs": None if args.pairs is None else str(args.pairs),
            "train_adapter": args.train_adapter, "tau_shift": str(args.tau_shift),
            "fake_updates": args.fake_updates, "accum": args.accum, "dmd_weight": args.dmd_weight, "betas": list(args.betas),
            "grad_clip": args.grad_clip, "ema": args.ema, "ema_warmup": args.ema > 0,
            "priming_weight": args.priming_weight, "priming_refs": args.priming_refs, "sigmas": run.sigmas,
            "paper_init": args.paper_init, "r_schedule": args.r_schedule, "head_lr": args.head_lr or args.lr}
    (args.out / "config.json").write_text(json.dumps(meta, indent=1))

    def save(tag: str) -> None:
        save_dir_run(args.out, tag, head=head, fake_head=fake_head, gen_network=gen_net, fake_network=fake_net,
                     adapter=adapter, meta=meta)
        if ema is not None:
            with ema.swapped():
                save_dir_run(args.out, f"{tag}_ema", head=head, fake_head=fake_head, gen_network=gen_net,
                             fake_network=fake_net, adapter=adapter, meta=meta, gen_only=True)
        print(f"LOT_H3 kind=dir_saved tag={tag or 'final'} out={args.out}", flush=True)

    @torch.no_grad()
    def evaluate(step: int) -> None:
        """Held-out prompts, fixed seeds. Per iteration: rms change from the previous estimate (does the
        refinement converge?) and the DMD terms at a fixed tau (|fake - real| mean, the normalised gap)."""
        head.eval()
        change = [[] for _ in range(args.iterations)]
        fake_real = [[] for _ in range(args.iterations)]
        hat_real = [[] for _ in range(args.iterations)]
        for index, item in enumerate(held_items):
            x0, text = train_h3.load_item(item, random.Random(1000 + index), device)
            gen = torch.Generator(device).manual_seed(index)
            z = torch.randn(x0.shape, device=device, generator=gen)
            eps = torch.randn(x0.shape, device=device, generator=gen)
            schedule = run.schedule(z)
            estimate, state = torch.zeros_like(z), z
            for r in range(args.iterations):
                with use_networks(gen_net, fake_net, gen=True, fake=False):
                    new = run.one_step(state, estimate, r, text, schedule, 7 * index)
                if r + 1 < args.iterations:
                    state = next_state(state, new, run.sigmas[r], run.sigmas[r + 1])
                change[r].append(float((new - estimate).pow(2).mean().sqrt()))
                tau = 0.6
                x_tau = (1.0 - tau) * new + tau * eps
                with use_networks(gen_net, fake_net, gen=False, fake=False):
                    real = real_x0(dit, x_tau, tau, text, audio_generator=run.audio(3 * index + r))
                with use_networks(gen_net, fake_net, gen=False, fake=True):
                    fake = fake_x0(dit, fake_head, x_tau, tau, r, text, audio_generator=run.audio(3 * index + r))
                fake_real[r].append(float((fake - real).abs().mean()))
                hat_real[r].append(float((new - real).abs().mean()))
                estimate = new
        head.train()
        mean = lambda xs: round(sum(xs) / max(len(xs), 1), 5)  # noqa: E731
        record = {"step": step, "eval": {"change": [mean(c) for c in change], "fake_real": [mean(c) for c in fake_real],
                                          "hat_real": [mean(c) for c in hat_real]}}
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(f"LOT_H3 kind=dir_eval step={step} change={record['eval']['change']} "
              f"fake_real={record['eval']['fake_real']} hat_real={record['eval']['hat_real']}", flush=True)

    if not args.resume_step:
        evaluate(0)
    started = time.perf_counter()
    for step in range(args.resume_step + 1, args.steps + 1):
        # One generator update = --accum samples (each its own prompt, noise, r and tau), grads
        # summed with 1/accum so the step matches a batch of that size. Batch 1 is the H3 runtime's limit.
        gen_opt.zero_grad(set_to_none=True)
        samples, all_stats = [], []
        for _ in range(args.accum):
            x0, text = train_h3.load_item(rng.choice(train_items), rng, device)
            z = torch.randn_like(x0)
            r = (step - 1) % args.iterations if args.r_schedule == "mod" else rng.randrange(args.iterations)
            run.loss_scale = 1.0 / args.accum
            _loss, x0_hat, stats = run.generator_update(z, text, r, real=x0)
            if not math.isfinite(stats["loss"]):
                raise RuntimeError(f"step {step}: generator loss {stats['loss']}")
            samples.append((z, text, r, x0_hat))
            all_stats.append(stats)
        run.loss_scale = 1.0
        grad_norm = clip(gen_params, args.grad_clip)
        gen_opt.step()
        if ema is not None:
            ema.update()
        gen_opt.zero_grad(set_to_none=True)
        # Fake updates: the first reuses the generator step's detached samples, the rest draw fresh
        # rollouts (Algorithm 1 draws fresh ones for every update; the reuse saves the rollouts).
        fake_losses = []
        for k in range(args.fake_updates):
            fake_opt.zero_grad(set_to_none=True)
            for z, text, r, x0_hat in samples:
                run.loss_scale = 1.0 / args.accum
                if k == 0:
                    floss, _tau = run.fake_update(z, text, r, x0_hat)
                else:
                    x1, text1 = train_h3.load_item(rng.choice(train_items), rng, device)
                    r1 = (step - 1) % args.iterations if args.r_schedule == "mod" else rng.randrange(args.iterations)
                    floss, _tau = run.fake_update(torch.randn_like(x1), text1, r1)
                fake_losses.append(float(floss.detach()))
            run.loss_scale = 1.0
            clip(fake_params, args.grad_clip)
            fake_opt.step()
            fake_opt.zero_grad(set_to_none=True)
        stats = {key: (sum(s[key] for s in all_stats) / len(all_stats)
                       if all(isinstance(s[key], (int, float)) for s in all_stats) else all_stats[-1][key])
                 for key in all_stats[0]}
        r = samples[-1][2]
        record = {"step": step, "r": [s[2] for s in samples], **stats, "grad_norm": float(grad_norm),
                  "fake_loss": sum(fake_losses) / max(len(fake_losses), 1),
                  "seconds": round(time.perf_counter() - started, 1)}
        log.write(json.dumps(record) + "\n")
        if step % 10 == 0 or step == args.steps:
            log.flush()
            print(f"LOT_H3 kind=dir step={step} r={r} dmd={stats['dmd']:.4g} fake_loss={record['fake_loss']:.4f} "
                  f"fake_real={stats['fake_real']:.4f} hat_real={stats['hat_real']:.4f} grad_norm={record['grad_norm']:.3g} "
                  f"s_per_step={(time.perf_counter() - started) / (step - args.resume_step):.2f} "
                  f"peak_mb={torch.cuda.max_memory_allocated() / 2**20:.0f}", flush=True)
        if args.eval_every and step % args.eval_every == 0:
            evaluate(step)
        if args.save_every and step % args.save_every == 0:
            save(f"_{step}")
    save("")
    print(f"LOT_H3 kind=dir_done steps={args.steps} seconds={time.perf_counter() - started:.0f} ok=1", flush=True)


if __name__ == "__main__":
    main()
