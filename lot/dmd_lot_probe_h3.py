#!/usr/bin/env python3
"""Probe: can DMD's real and fake models run on LoT layouts without bending the generator's gradient?

    python3 lot/dmd_lot_probe_h3.py --run lot/runs/dir_paper_run1 --step 100 --lot DIR_WITH_run4

DiR-DMD spends most of each update on the real and fake models at a random noise level tau,
and tau ~ Beta(2.5, 1) puts most of it at high noise, where the score is low-frequency: LoT's
comfort zone. If both models can run on coarse tokens there, training gets cheaper. This
script measures whether that changes the signal the generator learns from, on a saved run:

* generator samples ``x0_hat`` (the run's generator, a random iteration r);
* per tau, the DMD direction ``g = (x0_fake - x0_real) / tau`` (the paper's w(tau) = 1) with
  both models dense, and with both on a LoT layout (the trained LoT adapter + its LoRA stacked
  on top, so the fake keeps its own LoRA and r-embedding);
* ``cos`` and ``norm`` ratio of the LoT direction against the dense one, from the same x_tau;
* the floor: ``cos`` between dense directions from two different ``eps`` draws. DMD already
  lives with that much variation per sample, so LoT error well inside it is harmless.

Also prints DiT ms per forward, dense vs LoT, at the probed shape. Writes nothing but
``--out`` (json). One GPU job at a time; stop training first.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_h3  # noqa: E402
from dir_h3 import load_dir_run, use_networks  # noqa: E402
from h3 import grid_layout, make_h3_adapter  # noqa: E402
from h3_musubi import forward_video, load_dit, make_lora  # noqa: E402
from h3_splice import LotSplice  # noqa: E402
from lot_paths import LOT_H3_STILLS_CACHE  # noqa: E402

TAUS = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95)


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True, help="which save of --run (lora_gen_N, ...)")
    parser.add_argument("--lot", type=Path, required=True, help="trained LoT run dir (adapter + lora.safetensors)")
    parser.add_argument("--layouts", default="uniform2,bands,center")
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--swap", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="json path (default: RUN/dmd_lot_probe_N.json)")
    args = parser.parse_args()

    from gpu_guard import refuse_if_busy

    refuse_if_busy("dmd_lot_probe_h3.py")
    device = torch.device("cuda")
    meta_cfg = json.loads((args.run / "config.json").read_text())
    train_h3.MAX_LATENT_HW = (48, 48)
    items, _held = train_h3.list_items(Path(meta_cfg["cache"][0]) if meta_cfg.get("cache") else LOT_H3_STILLS_CACHE,
                                       int(meta_cfg.get("holdout", 8)))
    rng = random.Random(args.seed)

    dit = load_dit(device=device, swap=args.swap)
    head, gen_net, meta = load_dir_run(args.run, dit, tag=f"_{args.step}", device=device)
    fake_net = make_lora(dit, int(meta["rank"]))
    info = fake_net.load_weights(str(args.run / f"lora_fake_{args.step}.safetensors"))
    if info.missing_keys:
        raise RuntimeError("fake LoRA is missing keys")
    fake_net.requires_grad_(False)
    fake_net.to(device=device, dtype=torch.bfloat16)
    from dir_h3 import DirFakeHead

    fake_head = DirFakeHead(int(meta["temb_dim"]), int(meta["iters"]))
    fake_head.load_state_dict(torch.load(args.run / f"dir_fake_head_{args.step}.pt", map_location="cpu", weights_only=True))
    fake_head.to(device).eval()

    adapter = make_h3_adapter()
    adapter.load_state_dict(train_h3.load_adapter_state(args.lot))
    adapter.float().to(device).eval()
    lot_net = make_lora(dit, 16)
    lot_net.load_weights(str(args.lot / "lora.safetensors"))
    lot_net.requires_grad_(False)
    lot_net.to(device=device, dtype=torch.bfloat16)

    def set_lot(on: bool) -> None:
        for module in lot_net.unet_loras:
            module.enabled = on

    def x0_pred(x0_hat, eps, tau, text, *, fake: bool, r: int, splice=None, audio_seed=0) -> torch.Tensor:
        """Real (fake=False) or fake clean estimate at tau, in x-space.

        Dense: from ``x_tau = (1 - tau) x0_hat + tau eps``. LoT: from ``y_tau = (1 - tau) to_y(x0_hat) + tau eps``,
        the state LoT was trained on (extent scales apply to the clean part only), mapped back with ``to_x``."""
        audio = torch.Generator(device).manual_seed(audio_seed)
        delta = fake_head.fake_r_embed[r] if fake else None
        set_lot(splice is not None)
        with use_networks(gen_net, fake_net, gen=False, fake=fake):
            if splice is None:
                x_tau = (1.0 - tau) * x0_hat + tau * eps
                out = forward_video(dit, x_tau, 1.0 - tau, text, video_temb_delta=delta, audio_generator=audio).float()
                return x_tau.float() + tau * out
            y_tau = (1.0 - tau) * splice.to_y(x0_hat) + tau * eps
            out = forward_video(dit, y_tau, 1.0 - tau, text, lot=splice, video_temb_delta=delta,
                                audio_generator=audio).float()
            return splice.to_x(y_tau.float() + tau * out)

    names = [n for n in args.layouts.split(",") if n]
    rows = []
    timing: dict[str, list] = {"dense": [], **{n: [] for n in names}}
    with torch.no_grad():
        for index in range(args.samples):
            x0, text = train_h3.load_item(rng.choice(items), rng, device)
            gen = torch.Generator(device).manual_seed(args.seed * 1000 + index)
            z = torch.randn(x0.shape, device=device, generator=gen)
            r = index % int(meta["iters"])
            estimate = torch.zeros_like(z)
            from dir_h3 import generator_step

            set_lot(False)
            with use_networks(gen_net, fake_net, gen=True, fake=False):
                for k in range(r + 1):
                    estimate = generator_step(dit, head, z, estimate, k, text,
                                              audio_generator=torch.Generator(device).manual_seed(index))
            x0_hat = estimate
            _b, _c, tt, hh, ww = z.shape
            splices = {n: LotSplice(adapter, grid_layout(n, tt, hh // 2, ww // 2), y_space=True) for n in names}
            for tau in TAUS:
                eps1 = torch.randn(z.shape, device=device, generator=gen)
                eps2 = torch.randn(z.shape, device=device, generator=gen)
                seed = 17 * index + int(tau * 100)

                def direction(eps, splice=None, clock=None):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    real = x0_pred(x0_hat, eps, tau, text, fake=False, r=r, splice=splice, audio_seed=seed)
                    fake = x0_pred(x0_hat, eps, tau, text, fake=True, r=r, splice=splice, audio_seed=seed)
                    torch.cuda.synchronize()
                    if clock is not None:
                        clock.append((time.perf_counter() - start) / 2 * 1000.0)
                    return (fake - real) / tau, real

                g1, real1 = direction(eps1, clock=timing["dense"])
                g2, _ = direction(eps2)
                row = {"sample": index, "r": r, "tau": tau, "floor_cos": cos(g1, g2),
                       "dense_norm": float(g1.norm())}
                for n in names:
                    gl, real_l = direction(eps1, splices[n], clock=timing[n])
                    row[f"{n}_cos"] = cos(g1, gl)
                    row[f"{n}_norm"] = float(gl.norm() / g1.norm().clamp(min=1e-8))
                    row[f"{n}_real_err"] = float((real_l - real1).pow(2).mean().sqrt() / real1.pow(2).mean().sqrt())
                rows.append(row)
                print("LOT_H3 kind=dmd_lot_probe " + " ".join(
                    f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

    summary = {}
    for tau in TAUS:
        sel = [row for row in rows if row["tau"] == tau]
        entry = {"floor_cos": sum(x["floor_cos"] for x in sel) / len(sel)}
        for n in names:
            for key in ("cos", "norm", "real_err"):
                entry[f"{n}_{key}"] = sum(x[f"{n}_{key}"] for x in sel) / len(sel)
        summary[str(tau)] = entry
        print(f"LOT_H3 kind=dmd_lot_probe_tau tau={tau} " + " ".join(f"{k}={v:.3f}" for k, v in entry.items()), flush=True)
    ms = {k: sorted(v)[len(v) // 2] for k, v in timing.items() if v}
    print("LOT_H3 kind=dmd_lot_probe_ms " + " ".join(f"{k}={v:.0f}" for k, v in ms.items())
          + f" shape={tuple(z.shape)} swap={args.swap} peak_mb={torch.cuda.max_memory_allocated() / 2**20:.0f} ok=1",
          flush=True)
    out = args.out or (args.run / f"dmd_lot_probe_{args.step}.json")
    out.write_text(json.dumps({"rows": rows, "summary": summary, "ms_per_forward": ms, "lot": str(args.lot)}, indent=1))


if __name__ == "__main__":
    main()
