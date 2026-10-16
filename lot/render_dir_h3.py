#!/usr/bin/env python3
"""Same-seed stills: dense 20-step H3 against DiR (data-space iteration) at 1..R iterations.

    python3 lot/render_dir_h3.py --run lot/runs/train_dir_h3 [--prompts 3] [--iters 4]
    python3 lot/render_dir_h3.py --untrained --prompts 1      # fresh zero head, no LoRA: a smoke

Rows are prompts, columns are ``dense20`` (``h3_musubi.sample``, generator LoRA off) then
``dir r=1 .. R``. Every column starts from the same noise ``z``. The DiR iterates are the
intermediate ``x0_hat[r]`` of ONE ``dir_sample`` run: no renoising and a fixed ``z`` make
iteration ``r`` independent of how many follow, so the r=1 column is what a one-step
generator would give. ``grid.png`` and ``meta.json`` go to ``--out`` (default
``lot/runs/render_dir``, gitignored). ``meta.json`` holds DiT milliseconds per sample and
``delta[r] = mean |x0_hat[r] - x0_hat[r-1]|`` (``x0_hat[0] = 0``): does refinement converge?

Run layout read by ``load_run`` (keep it the one place to adapt to the trainer):
``dir_head.pt`` (``DirHead`` state_dict), ``lora_gen.safetensors`` (rank in its metadata key
``rank``, else ``config.json``), optional ``adapter.pt`` (needed when the schedule has a
non-dense layout), ``config.json`` with ``schedule``, ``iters``, ``rank``, ``pairs``. The fake
model's files are not needed to sample. ``--untrained`` is a smoke test, not a quality claim:
a zero head makes every iteration the one-step base prediction (up to bf16 rounding). Smoke-tested
on the GPU with a 2-step run (``--run``, with and without ``--ema``) and ``--untrained``; no
trained weights worth judging exist yet. ``--ema`` loads the ``*_ema`` files; ``--holdout`` and
``--prompt-cache`` default to the run's ``config.json``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before CUDA init

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lot_paths import LOT_H3_CHECKPOINT, LOT_H3_VAE  # noqa: E402

from gpu_guard import refuse_if_busy  # noqa: E402
from render_h3 import HEIGHT as _H, WIDTH as _W, grid, load_prompts, to_image  # noqa: E402

HEIGHT, WIDTH = _H, _W   # --size overrides

DEFAULT_R = 4


def read_config(run: Path | None) -> dict:
    path = None if run is None else run / "config.json"
    return json.loads(path.read_text()) if path is not None and path.is_file() else {}


def load_run(run: Path, dit, config: dict, rank: int | None, ema: bool = False, tag: str = ""):
    """``(head, gen_network, adapter_state)`` of a trained run. The one function to adapt to the trainer."""
    from safetensors import safe_open

    from dir_h3 import DirHead, probe_temb_dim
    from h3_musubi import make_lora

    sfx = tag + ("_ema" if ema else "")
    lora_path = run / f"lora_gen{sfx}.safetensors"
    if not lora_path.is_file():
        raise SystemExit(f"{lora_path} not found" + (" (the run was trained without --ema)" if ema else ""))
    with safe_open(str(lora_path), "pt") as handle:
        meta_rank = (handle.metadata() or {}).get("rank")
    rank = rank or (int(meta_rank) if meta_rank else None) or config.get("rank")
    if not rank:
        raise SystemExit("LoRA rank unknown: no `rank` in the safetensors metadata or config.json; pass --rank")
    head = DirHead(dit.video_patch_proj.out_features, probe_temb_dim(dit), int(config.get("iters", DEFAULT_R)))
    head.load_state_dict(torch.load(run / f"dir_head{sfx}.pt", map_location="cpu"))
    network = make_lora(dit, int(rank))
    info = network.load_weights(str(lora_path))
    if info.missing_keys:
        raise RuntimeError(f"LoRA load missed {len(info.missing_keys)} keys")
    network.to(device="cuda", dtype=torch.bfloat16).eval()
    adapter_state = None
    if (run / f"adapter{sfx}.pt").is_file():
        adapter_state = torch.load(run / f"adapter{sfx}.pt", map_location="cpu", weights_only=True)
    elif not ema and (run / "adapter.safetensors").is_file():
        from train_h3 import load_adapter_state

        adapter_state = load_adapter_state(run)
    return head.cuda().eval(), network, adapter_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path, help="train_dir_h3 output dir")
    source.add_argument("--untrained", action="store_true", help="zero DirHead, no LoRA (smoke test)")
    parser.add_argument("--prompts", type=int, default=3)
    parser.add_argument("--iters", type=int, default=None, help="R (default: config.json iters, else 4)")
    parser.add_argument("--schedule", default=None,
                        help="layout per iteration, e.g. bands,bands,dense,dense (default: config.json, else dense)")
    parser.add_argument("--sigmas", default=None,
                        help="flow sigma per iteration: 1, grid[:SHIFT], turbo[:SHIFT] or a comma list (default: config.json sigmas, else 1)")
    parser.add_argument("--dense-from", type=Path, default=None,
                        help="reuse the dense column of an earlier render (same prompts, seed, canvas, steps) instead of sampling it")
    parser.add_argument("--steps", type=int, default=20, help="dense baseline steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "runs" / "render_dir")
    parser.add_argument("--swap", type=int, default=0, help="blocks streamed from CPU (0 = all resident)")
    parser.add_argument("--rank", type=int, default=None, help="gen LoRA rank if the file and config.json lack it")
    parser.add_argument("--size", type=int, nargs=2, default=None, metavar=("W", "H"),
                        help="canvas in pixels, multiples of 32 (default 768 1152); training crops were up to 768x768")
    parser.add_argument("--tag", default="", help="checkpoint suffix, e.g. _600 for the step-600 save (default: final)")
    parser.add_argument("--ema", action="store_true", help="load the *_ema generator files (run trained with --ema)")
    parser.add_argument("--holdout", type=int, default=None,
                        help="the --holdout the trainer used for that cache (default: config.json holdout, else 24)")
    parser.add_argument("--pick", default=None, help="comma list of held-out indices to render, e.g. 7,3")
    parser.add_argument("--prompt-cache", type=Path, default=None,
                        help="with --run: held-out prompts from this cache (default: config.json cache[0], else train_h3's cache_iso3d)")
    args = parser.parse_args()
    global HEIGHT, WIDTH
    if args.size:
        WIDTH, HEIGHT = args.size

    refuse_if_busy("render_dir_h3.py")
    from dir_h3 import DirHead, dir_sample, parse_layout_schedule, parse_sigmas, probe_temb_dim, use_networks
    from h3 import make_h3_adapter
    from h3_musubi import load_dit, load_vae, sample

    config = read_config(args.run)
    trained_iters = int(config.get("iters", DEFAULT_R))
    iters = args.iters or trained_iters
    if args.run is not None and not 1 <= iters <= trained_iters:
        raise SystemExit(f"--iters {iters} is outside 1..{trained_iters}: the run trained {trained_iters} iteration embeddings")
    spec = args.schedule or config.get("schedule") or "dense"
    schedule = parse_layout_schedule(spec, iters, HEIGHT // 32, WIDTH // 32, 1)
    # a run's sigmas cover its trained R; rendering fewer iterations uses the first ones
    sigmas = parse_sigmas(args.sigmas or config.get("sigmas") or "1",
                          trained_iters if args.run is not None else iters)[:iters]

    if args.run is not None and any(n.strip() != "dense" for n in spec.split(",")) \
            and not ((args.run / "adapter.pt").is_file() or (args.run / "adapter.safetensors").is_file()):
        print(f"WARNING: LoT schedule {spec!r} but {args.run} has no adapter.pt: the adapter is the untrained "
              f"mean lift, not what the run trained with", flush=True)

    model = load_dit(LOT_H3_CHECKPOINT, swap=args.swap)
    network, adapter, adapter_state = None, None, None
    if args.untrained:
        head = DirHead(model.video_patch_proj.out_features, probe_temb_dim(model), iters).cuda().eval()
    else:
        head, network, adapter_state = load_run(args.run, model, config, args.rank, args.ema, args.tag)
    if any(layout is not None for layout in schedule):
        adapter = make_h3_adapter()
        if adapter_state is not None:
            adapter.load_state_dict(adapter_state)
        else:
            from procrustes_h3 import CHECKPOINT, pretrained_maps

            adapter.init_from_pretrained(*pretrained_maps(CHECKPOINT))   # untrained: mean lift
        adapter = adapter.cuda().eval()

    held = None
    if args.run is not None:
        from train_h3 import CACHE, list_items

        cache = args.prompt_cache or (Path(config["cache"][0]) if config.get("cache") else CACHE)
        holdout = args.holdout if args.holdout is not None else int(config.get("holdout", 24))
        held = [te for _latent, te in list_items(cache, holdout)[1]]
        if args.pick:
            held = [held[int(index)] for index in args.pick.split(",")]
    prompts = load_prompts(args.prompts, held)
    if args.dense_from is not None:
        old = json.loads((args.dense_from / "meta.json").read_text())
        want = {"prompts": [text for text, _ in prompts][:len(old["prompts"])], "seed": args.seed,
                "canvas": f"{WIDTH}x{HEIGHT}", "steps": args.steps}
        have = {"prompts": old["prompts"][:len(prompts)], "seed": old["seed"], "canvas": old["canvas"],
                "steps": old.get("steps", 20)}
        if want != have or len(old["prompts"]) < len(prompts):
            raise SystemExit(f"--dense-from {args.dense_from}: prompts/seed/canvas/steps differ")

    latents: dict[tuple[int, str], torch.Tensor] = {}
    dense_ms, dir_ms, deltas = [], [], []
    for index, (_text, embeds) in enumerate(prompts):
        embeds = embeds.to("cuda", torch.bfloat16)
        seed = args.seed + index
        torch.cuda.synchronize()
        start = time.perf_counter()
        if args.dense_from is None:
            with torch.inference_mode(), use_networks(network, None, gen=False, fake=False):
                dense = sample(model, embeds, width=WIDTH, height=HEIGHT, steps=args.steps, seed=seed)
            torch.cuda.synchronize()
            dense_ms.append(1000 * (time.perf_counter() - start))
            latents[(index, "dense20")] = dense.float().cpu()
        else:
            dense_ms.append(float(old["dit_ms"]["dense"][index]))

        start = time.perf_counter()
        with torch.inference_mode(), use_networks(network, None, gen=True, fake=False):
            _final, estimates = dir_sample(model, head, embeds, height=HEIGHT, width=WIDTH, iters=iters,
                                           seed=seed, schedule=schedule, adapter=adapter, sigmas=sigmas)
        torch.cuda.synchronize()
        dir_ms.append(1000 * (time.perf_counter() - start))
        previous = torch.zeros_like(estimates[0])
        row = []
        for r, estimate in enumerate(estimates, 1):
            latents[(index, f"dir{r}")] = estimate.float().cpu()
            row.append(float((estimate.float() - previous.float()).abs().mean()))
            previous = estimate
        deltas.append(row)
        print(f"LOT_H3 kind=render_dir prompt={index} dense_ms={dense_ms[-1]:.0f} dir_ms={dir_ms[-1]:.0f} "
              f"delta={','.join(f'{d:.4f}' for d in row)} "
              f"finite={int(all(bool(torch.isfinite(estimate).all()) for estimate in estimates))}", flush=True)
    del model, head, network, adapter
    gc.collect()
    torch.cuda.synchronize()
    decoder = load_vae(LOT_H3_VAE)

    args.out.mkdir(parents=True, exist_ok=True)
    names = ["dense20"] + [f"dir{r}" for r in range(1, iters + 1)]
    rows = []
    for index in range(len(prompts)):
        row = []
        for name in names:
            if name == "dense20" and args.dense_from is not None:
                from PIL import Image

                image = Image.open(args.dense_from / f"p{index}_dense20.png").convert("RGB")
            else:
                with torch.inference_mode():
                    image = to_image(decoder.decode(latents[(index, name)].to("cuda", next(decoder.parameters()).dtype)))
            image.save(args.out / f"p{index}_{name}.png")
            row.append(image)
        rows.append(row)
    dense_med, dir_med = float(np.median(dense_ms)), float(np.median(dir_ms))
    labels = [f"dense{args.steps}  {dense_med:.0f}ms"] + [
        f"dir r={r}  ~{dir_med * r / iters:.0f}ms" for r in range(1, iters + 1)]
    grid(rows, labels).save(args.out / "grid.png")
    meta = {
        "run": None if args.run is None else str(args.run), "untrained": bool(args.untrained), "ema": bool(args.ema),
        "iters": iters, "schedule": spec, "sigmas": sigmas,
        "dense_from": None if args.dense_from is None else str(args.dense_from), "steps": args.steps, "seed": args.seed, "canvas": f"{WIDTH}x{HEIGHT}",
        "prompts": [text for text, _ in prompts],
        "dit_ms": {"dense": dense_ms, "dir_total": dir_ms, "dense_median": dense_med, "dir_median": dir_med,
                   "dir_per_iter_median": dir_med / iters,
                   "note": "dir r=k column cost is k/R of dir_total (iterations are timed together; "
                           "the first prompt includes CUDA warm-up)"},
        "delta": {"definition": "mean |x0_hat[r] - x0_hat[r-1]|, x0_hat[0] = 0; one row per prompt, r = 1..R",
                  "per_prompt": deltas},
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"LOT_H3 kind=render_dir_done out={args.out} iters={iters} dense_ms={dense_med:.0f} "
          f"dir_ms={dir_med:.0f} ok=1", flush=True)


if __name__ == "__main__":
    main()
