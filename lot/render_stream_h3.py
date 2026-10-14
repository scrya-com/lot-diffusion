#!/usr/bin/env python3
"""Streaming DiR video: one sliding window, refined and never re-noised (``stream_dir_h3``).

    python3 lot/render_stream_h3.py --run lot/runs/dir_paper_run1 --tag _1000 --clips 4
    python3 lot/render_stream_h3.py --untrained --clips 2 --prompts 1        # zero head, no LoRA: a smoke

``--run`` is a ``train_stream_dir_h3.py`` run, or a still-image DiR run (``train_dir_h3.py``)
warm-started into a ``StreamHead`` (``load_stream_run``): its LoRA and iteration embeddings are
reused, the done row and the memory projection are zero. A still run shows what the still
generator does with per-frame ``r``, the FL2VA ``first`` condition and the coarse memory, not a
trained stream.

``--first``: ``dense`` renders the opening frame with the dense 20-step H3 at one latent frame (a
one-pixel-frame latent, exactly what the FL2VA ``first`` condition is); ``none`` starts the first
window from noise (T2VA). Each window runs ``R`` forwards and emits one VAE clip (17 pixel frames).

Writes ``p{i}.mp4`` (24 fps), ``p{i}_sheet.png`` (every 8th frame) and ``meta.json`` (per-forward
DiT milliseconds, per-clip mean change) to ``--out`` (default ``lot/runs/render_stream``, gitignored).
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

from devices import add_device_arg, empty_cache, pick_device, swap_for, synchronize  # noqa: E402
from gpu_guard import refuse_if_busy  # noqa: E402
from render_h3 import load_prompts  # noqa: E402

FPS = 24


def frames_of(pixels: torch.Tensor) -> np.ndarray:
    """``(1, 3, F, H, W)`` in [-1, 1] to ``(F, H, W, 3)`` uint8."""
    rgb = (pixels[0].permute(1, 2, 3, 0).float().cpu().numpy() + 1.0) * 0.5
    return (rgb * 255.0).round().clip(0, 255).astype(np.uint8)


def sheet(frames: np.ndarray, every: int = 8, scale: float = 0.25):
    from PIL import Image

    picks = [Image.fromarray(frame) for frame in frames[::every]]
    width, height = int(picks[0].width * scale), int(picks[0].height * scale)
    out = Image.new("RGB", (width * len(picks), height), "white")
    for index, image in enumerate(picks):
        out.paste(image.resize((width, height), Image.LANCZOS), (index * width, 0))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path,
                        help="train_stream_dir_h3 run, or a train_dir_h3 (still) run to warm-start from")
    source.add_argument("--untrained", action="store_true", help="zero StreamHead, no LoRA (smoke test)")
    parser.add_argument("--tag", default="", help="checkpoint suffix, e.g. _1000 (default: final)")
    parser.add_argument("--ema", action="store_true", help="load the *_ema generator files")
    parser.add_argument("--prompts", type=int, default=2)
    parser.add_argument("--clips", type=int, default=4, help="VAE clips to stream (17 pixel frames each)")
    parser.add_argument("--frames", type=int, default=7, help="window latent frames (5n+2)")
    parser.add_argument("--size", type=int, nargs=2, default=[768, 768], metavar=("W", "H"),
                        help="pixels, multiples of 32 (the still run trained on crops up to 768x768)")
    parser.add_argument("--first", choices=["dense", "none"], default="dense")
    parser.add_argument("--steps", type=int, default=20, help="dense steps for --first dense")
    parser.add_argument("--pool", type=int, default=4, help="memory: coarse tokens average pool x pool patch tokens")
    parser.add_argument("--capacity", type=int, default=16, help="memory: keyframes kept (one per clip)")
    parser.add_argument("--memory-time", choices=["real", "packed"], default="real")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--swap", type=int, default=0, help="blocks streamed from CPU (0 = all resident; CUDA only)")
    add_device_arg(parser)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "runs" / "render_stream")
    args = parser.parse_args()

    refuse_if_busy("render_stream_h3.py")
    device = pick_device(args.device)
    args.swap = swap_for(device, args.swap)
    from dir_h3 import probe_temb_dim, use_networks
    from h3_musubi import load_dit, load_vae, sample
    from stream_dir_h3 import StreamHead, check_window, load_stream_run, stream_sample

    check_window(args.frames)
    width, height = args.size
    if width % 32 or height % 32:
        raise SystemExit("--size must be multiples of 32")
    shape = (args.frames, height // 16, width // 16)

    model = load_dit(LOT_H3_CHECKPOINT, device=device, swap=args.swap)
    temb_dim, hidden = probe_temb_dim(model), model.config.hidden_size
    network, config = None, {}
    if args.untrained:
        head = StreamHead(hidden, temb_dim).to(device).eval()
    else:
        head, network, config = load_stream_run(args.run, model, tag=args.tag, device=device, ema=args.ema)
    held = None
    if args.run is not None:
        from train_h3 import CACHE, list_items

        cache = Path(config["cache"][0]) if config.get("cache") else CACHE
        held = [te for _latent, te in list_items(cache, int(config.get("holdout", 8)))[1]]
    prompts = load_prompts(args.prompts, held)

    results = []
    for index, (caption, embeds) in enumerate(prompts):
        embeds = embeds.to(device, torch.bfloat16)
        seed = args.seed + index
        first, first_ms = None, 0.0
        if args.first == "dense":
            synchronize(device)
            start = time.perf_counter()
            with torch.inference_mode(), use_networks(network, None, gen=False, fake=False):
                first = sample(model, embeds, width=width, height=height, steps=args.steps, seed=seed,
                               device=device)
            synchronize(device)
            first_ms = 1000 * (time.perf_counter() - start)
        forward_ms, clip_change = [], []
        clip_start = [time.perf_counter()]

        def on_clip(clip_index, clip):
            synchronize(device)
            now = time.perf_counter()
            forward_ms.append(1000 * (now - clip_start[0]))
            clip_start[0] = now
            clip_change.append(float(clip.float().abs().mean()))

        with torch.inference_mode(), use_networks(network, None, gen=True, fake=False):
            latents = stream_sample(model, head, embeds, shape=shape, clips=args.clips, first=first, seed=seed,
                                    pool=args.pool, capacity=args.capacity, memory_time=args.memory_time,
                                    device=device, on_clip=on_clip, tail=True)
        finite = bool(torch.isfinite(latents).all())
        results.append({"caption": caption, "latents": latents.float().cpu(), "first_ms": first_ms,
                        "clip_ms": forward_ms, "clip_abs_mean": clip_change, "finite": finite})
        print(f"LOT_H3 kind=render_stream prompt={index} first_ms={first_ms:.0f} "
              f"clip_ms={','.join(f'{ms:.0f}' for ms in forward_ms)} finite={int(finite)}", flush=True)

    del model, head, network
    gc.collect()
    synchronize(device)
    empty_cache(device)
    decoder = load_vae(LOT_H3_VAE, device)
    dtype = next(decoder.parameters()).dtype
    args.out.mkdir(parents=True, exist_ok=True)
    import imageio

    for index, result in enumerate(results):
        with torch.inference_mode():
            pixels = decoder.decode(result["latents"].to(device, dtype))
        frames = frames_of(pixels)
        imageio.mimwrite(args.out / f"p{index}.mp4", list(frames), fps=FPS, quality=8, macro_block_size=16)
        sheet(frames).save(args.out / f"p{index}_sheet.png")
        result["pixel_frames"] = int(frames.shape[0])
    iters = 4 if args.untrained else int(config.get("iters", 4))
    meta = {"run": None if args.run is None else str(args.run), "tag": args.tag, "ema": args.ema,
            "untrained": args.untrained, "size": f"{width}x{height}", "window": list(shape), "clips": args.clips,
            "first": args.first, "pool": args.pool, "capacity": args.capacity, "memory_time": args.memory_time,
            "iters": iters, "fps": FPS, "seed": args.seed,
            "note": "clip_ms is wall time per emitted clip (R window forwards, 17 pixel frames, "
                    f"{17 / FPS:.3f} s of video); the first clip includes warm-up",
            "prompts": [{k: v for k, v in r.items() if k != "latents"} for r in results]}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"LOT_H3 kind=render_stream_done out={args.out} ok=1", flush=True)


if __name__ == "__main__":
    main()
