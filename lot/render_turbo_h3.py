#!/usr/bin/env python3
"""Minimax-H3-Turbo (LightX2V) LoRAs on the same held-out prompts and seeds as a render_dir_h3 run.

    python3 lot/render_turbo_h3.py --dir-render lot/runs/render_dir_1000_live_full --run lot/runs/dir_paper_run1 \
        --lora 4:/path/turbo_4step_768p_comfyui.safetensors --lora 8:/path/turbo_8step_768p_comfyui.safetensors

Turbo's ComfyUI LoRAs name the same modules musubi H3 has (blocks.N.attn.qkv_proj fused, out_proj,
mlp.fc1 as [gate; value], fc2, plus the 2 token-refiner blocks). They are applied as forward hooks,
``y += (alpha / rank) * B(A(x))``, so the int8 base is untouched. Sampling follows Turbo's
inference script: N NFE on the unshifted grid q = linspace(1, 0, N + 1), video sigma =
shift(q, --video-shift), audio sigma = shift(q, 3), and the same Euler update as
``h3_musubi.sample`` (diffusers' MiniMaxH3Scheduler). The noise is seeded exactly like
``render_dir_h3`` (seed + prompt index), so the columns line up with that run's dense and DiR images.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

LOT = Path(__file__).resolve().parent
sys.path.insert(0, str(LOT))
from lot_paths import LOT_H3_CHECKPOINT, LOT_H3_VAE  # noqa: E402

from gpu_guard import refuse_if_busy  # noqa: E402


def hook_lora(dit, path: Path, device):
    from safetensors import safe_open

    handles, count = [], 0
    with safe_open(str(path), "pt") as f:
        names = sorted({k.rsplit(".", 2)[0] for k in f.keys() if k.endswith("lora_A.weight")})
        for name in names:
            module = dit.get_submodule(name.removeprefix("diffusion_model."))
            a = f.get_tensor(f"{name}.lora_A.weight").to(device, torch.bfloat16)
            b = f.get_tensor(f"{name}.lora_B.weight").to(device, torch.bfloat16)
            scale = float(f.get_tensor(f"{name}.alpha")) / a.shape[0]

            def hook(mod, inputs, output, a=a, b=b, scale=scale):
                x = inputs[0]
                delta = (x.to(torch.bfloat16) @ a.T) @ b.T
                return output + (scale * delta).to(output.dtype)

            handles.append(module.register_forward_hook(hook))
            count += 1
    return handles, count


@torch.no_grad()
def turbo_sample(dit, text, *, height, width, nfe, video_shift, seed, device="cuda"):
    from h3_musubi import _layout, _silence, _sync_swap, shift_sigma

    generator = torch.Generator(device).manual_seed(seed)
    x = torch.randn(1, 24, 1, height // 16, width // 16, device=device, generator=generator)
    audio_generator = torch.Generator(device).manual_seed(seed + 1)
    q = torch.linspace(1.0, 0.0, nfe + 1, dtype=torch.float64)
    sv, sa = shift_sigma(q, video_shift), shift_sigma(q, 3.0)
    layout = _layout(text.shape[1], 1, x.shape[-2], x.shape[-1])
    tags = torch.ones(1, text.shape[1], dtype=torch.int64)
    for i in range(nfe):
        _sync_swap(dit)
        audio = _silence(1, float(sa[i]), device, audio_generator)
        v = dit(video_latents=x, audio_latents=audio, text_hidden_states=text, text_token_tags=tags, layout=layout,
                model_t_video=1.0 - float(sv[i]), model_t_audio=1.0 - float(sa[i])).video.float()
        x0 = x + float(sv[i]) * v
        ratio = float(sv[i + 1]) / float(sv[i])
        x = ratio * x + (1.0 - ratio) * x0
    return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir-render", type=Path, required=True, help="a render_dir_h3 output dir (meta.json, p*_*.png)")
    parser.add_argument("--run", type=Path, required=True, help="the DiR run that render used (prompts: holdout + cache)")
    parser.add_argument("--lora", action="append", required=True, help="NFE:path, e.g. 4:/x/turbo_4step.safetensors")
    parser.add_argument("--video-shift", type=float, default=6.0, help="Turbo 768p LoRAs: 6")
    parser.add_argument("--swap", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    refuse_if_busy("render_turbo_h3.py")
    from h3_musubi import load_dit, load_vae
    from render_h3 import load_prompts, to_image
    from train_h3 import CACHE, list_items

    meta = json.loads((args.dir_render / "meta.json").read_text())
    width, height = (int(v) for v in meta["canvas"].split("x"))
    config = json.loads((args.run / "config.json").read_text())
    cache = Path(config["cache"][0]) if config.get("cache") else CACHE
    held = [te for _latent, te in list_items(cache, int(config.get("holdout", 24)))[1]]
    prompts = load_prompts(len(meta["prompts"]), held)
    if [p for p, _ in prompts] != meta["prompts"]:
        raise SystemExit("prompt list differs from the DiR render's meta.json")

    dit = load_dit(LOT_H3_CHECKPOINT, swap=args.swap)
    latents, timings = {}, {}
    for spec in args.lora:
        nfe, path = spec.split(":", 1)
        nfe = int(nfe)
        handles, count = hook_lora(dit, Path(path), "cuda")
        print(f"LOT_H3 kind=turbo_lora nfe={nfe} modules={count} path={path}", flush=True)
        timings[nfe] = []
        for index, (_caption, embeds) in enumerate(prompts):
            embeds = embeds.to("cuda", torch.bfloat16)
            torch.cuda.synchronize()
            start = time.perf_counter()
            x = turbo_sample(dit, embeds, height=height, width=width, nfe=nfe, video_shift=args.video_shift,
                             seed=int(meta["seed"]) + index)
            torch.cuda.synchronize()
            timings[nfe].append(1000 * (time.perf_counter() - start))
            latents[(index, nfe)] = x.float().cpu()
            print(f"LOT_H3 kind=turbo prompt={index} nfe={nfe} ms={timings[nfe][-1]:.0f} "
                  f"finite={int(bool(torch.isfinite(x).all()))}", flush=True)
        for h in handles:
            h.remove()
        del handles
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({f"{i}_{n}": v for (i, n), v in latents.items()}, args.out / "latents.pt")   # survive a decode OOM
    import gc

    del dit
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print(f"LOT_H3 kind=turbo_freed allocated_mb={torch.cuda.memory_allocated() / 2**20:.0f}", flush=True)
    decoder = load_vae(LOT_H3_VAE)
    dtype = next(decoder.parameters()).dtype
    args.out.mkdir(parents=True, exist_ok=True)

    from PIL import Image, ImageDraw

    nfes = sorted(timings)
    cols = ["dense20", "dir1", "dir4"] + [f"turbo{n}" for n in nfes]
    dense_ms = float(np.median(meta["dit_ms"]["dense"]))
    dir_iter = float(meta["dit_ms"]["dir_per_iter_median"])
    labels = {"dense20": f"dense 20 steps  {dense_ms / 1000:.1f}s", "dir1": f"DiR r=1 (step 1000)  {dir_iter / 1000:.1f}s",
              "dir4": f"DiR r=4 (step 1000)  {4 * dir_iter / 1000:.1f}s"}
    for n in nfes:
        labels[f"turbo{n}"] = f"Turbo {n}-step 768p  {float(np.median(timings[n])) / 1000:.1f}s"
    scale = 0.33
    cw, ch, header = int(width * scale), int(height * scale), 28
    sheet = Image.new("RGB", (cw * len(cols), header + ch * len(prompts)), "white")
    draw = ImageDraw.Draw(sheet)
    for c, name in enumerate(cols):
        draw.text((c * cw + 6, 8), labels[name], fill="black")
    for r in range(len(prompts)):
        for c, name in enumerate(cols):
            if name.startswith("turbo"):
                with torch.inference_mode():
                    image = to_image(decoder.decode(latents[(r, int(name[5:]))].to("cuda", dtype)))
                image.save(args.out / f"p{r}_{name}.png")
            else:
                image = Image.open(args.dir_render / f"p{r}_{name}.png").convert("RGB")
            sheet.paste(image.resize((cw, ch), Image.LANCZOS), (c * cw, header + r * ch))
    sheet.save(args.out / "grid.png")
    out_meta = {"dir_render": str(args.dir_render), "loras": args.lora, "video_shift": args.video_shift,
                "audio_shift": 3.0, "canvas": meta["canvas"], "seed": meta["seed"], "prompts": meta["prompts"],
                "dit_ms": {"dense_median": dense_ms, "dir_per_iter_median": dir_iter,
                           **{f"turbo{n}": timings[n] for n in nfes}}}
    (args.out / "meta.json").write_text(json.dumps(out_meta, indent=2) + "\n")
    print(f"LOT_H3 kind=turbo_done out={args.out} ok=1", flush=True)


if __name__ == "__main__":
    main()
