#!/usr/bin/env python3
"""DRAFT: ``L`` nested resolution levels on MiniMax H3 (generalises ``two_track_h3.py``).

    python3 lot/pyramid_h3.py plan --size 2688 1536          # CPU: levels, tokens, attention cost
    python3 lot/pyramid_h3.py render --size 2688 1536 --out DIR --boxes 0.2 0.1 0.8 0.95 0.35 0.15 0.65 0.5

Paper: MPEG-Diffusion: Mixed-Resolution Video Generation, Dennis Menn, Tianpei Gu, Guande He,
Aryan V S, Yanke Song, Nan Liu, Desai Xie, Xun Huang, Chenfeng Xu and Diana Marculescu (UT Austin,
Roblox), arXiv:2611.12069. The paper uses two levels; this file is our extension, see
``lot/pyramid.py`` for what more levels do and do not buy.

Sampler (draft): level 0 (the whole frame at 1/2^(L-1) resolution) denoises alone until
``--switches[0]``. At each switch the next level starts on a nested box: the composite clean
estimate is upscaled through the VAE, re-noised at the switch sigma, and from then on all existing
levels denoise together in one sequence with ``pyramid.pyramid_attention``. The output is built
coarse to fine: upscale, paste the next level's region, repeat.

Status: ``plan`` runs on CPU. ``render`` is wired but has not been run on a GPU; there are no
fitted maps for more than one level step and no training. Untested assumptions are marked TODO.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pyramid import PyramidPlan, attention_cost, max_levels, pyramid_attention, token_share  # noqa: E402

TOKEN_PX = 32          # H3: one full-resolution token is 2x2 latent pixels of 16 px


def pyramid_forward(dit, latents: list[torch.Tensor], plan: PyramidPlan, t: float, text: torch.Tensor, *,
                    maps=None, audio_generator=None) -> list[torch.Tensor]:
    """Velocities of every level's latent (zero where a level has no tokens). ``latents[l]`` is the
    complete level-``l`` latent ``(1, 24, T, h * 2^l * 2, w * 2^l * 2)``; only its existing tokens
    enter the sequence. No-grad draft: training needs the persistent patch of ``two_track_h3``."""
    from h3_musubi import _layout, _silence, _sync_swap, audio_sigma
    from mixed_res import patched_attention
    from musubi_tuner.minimax_h3.packing import (H3RowSegment, _unpack_video_rows, build_position_grid,
                                                 build_timestep_rows, pack_audio_rows, pack_video_rows)

    if text.ndim == 2:
        text = text.unsqueeze(0)
    _sync_swap(dit)
    device = dit.video_patch_proj.weight.device
    sigma_a = float(audio_sigma(1.0 - t))
    frames = latents[0].shape[2]
    layouts = [_layout(text.shape[1], frames, lat.shape[-2], lat.shape[-1]) for lat in latents]
    segment = layouts[0].target_video_segment
    rows, positions, grid_tables = [], [build_position_grid(layouts[0])[0]], []
    for level, (latent, layout) in enumerate(zip(latents, layouts)):
        packed = pack_video_rows(latent.to(device, torch.float32))
        if level == 0:
            rows.append(packed)
            continue
        grid = build_position_grid(layout)[0][-plan.grid_counts[level]:]
        rows.append(packed[:, plan.index[level]])
        positions.append(grid[plan.index[level].cpu()])
        grid_tables.append(dit._rotation_table(grid.unsqueeze(0).to(device), dit.dtype))
    video = dit.video_patch_proj(torch.cat(rows, dim=1)).to(dit.dtype)
    audio = _silence(frames, sigma_a, device, audio_generator)
    audio_rows = dit.audio_patch_proj(pack_audio_rows(audio.to(device)).to(torch.float32)).to(dit.dtype)
    text_rows = dit.token_refiner(dit.condition_proj(text.to(device, dit.dtype)))
    hidden = torch.cat([text_rows, audio_rows, video], dim=1)
    total = sum(plan.counts)
    base = layouts[0]
    packed_layout = replace(base, segments=base.segments[:-1] + (
        H3RowSegment(segment.role, segment.kind, segment.start, segment.start + total),), row_count=segment.start + total)
    # positions[0] is the whole level-0 packed grid (text, audio, video); later entries are video rows.
    rotation = dit._rotation_table(torch.cat(positions).unsqueeze(0).to(device), hidden.dtype)
    tags = torch.ones(1, text.shape[1], dtype=torch.int64)
    timesteps = build_timestep_rows(packed_layout, text_token_tags=tags, model_t_video=t, model_t_audio=1.0 - sigma_a)
    temb = dit._timestep_embeddings(timesteps.unique_timesteps, device)
    patch = patched_attention(dit.blocks, attention=pyramid_attention, plan=plan, maps=maps,
                              video_start=segment.start, grid_rotations=grid_tables) if plan.levels > 1 \
        else contextlib.nullcontext()
    with patch:
        for index, block in enumerate(dit.blocks):
            if dit.blocks_to_swap:
                dit.offloader.wait_for_block(index)
            hidden = block(hidden, temb, timesteps.block_segments, rotation)
            if dit.blocks_to_swap:
                dit.offloader.submit_move_blocks_forward(dit.blocks, index)
    final = dit.final_layer
    shift, scale = final.adaln_proj(temb)
    step = timesteps.video_timestep_index
    out = final.video_out((final.norm(hidden.to(device)[:, segment.start:]) * (1.0 + scale[step]) + shift[step])
                          .to(torch.float32))
    velocities, row = [], 0
    for level, (latent, layout) in enumerate(zip(latents, layouts)):
        full = out.new_zeros(1, plan.grid_counts[level], out.shape[-1])
        full[:, plan.index[level]] = out[:, row:row + plan.counts[level]]
        row += plan.counts[level]
        velocities.append(_unpack_video_rows(full, layout.target_video).to(latent.dtype))
    return velocities


def compose(latents: list[torch.Tensor], masks: list[torch.Tensor], upscale) -> torch.Tensor:
    """Coarse to fine: upscale the composite so far, paste the next level inside its region. A level-``l``
    region (a mask on level ``l - 1`` tokens) covers 4x4 latent pixels per token at level ``l``."""
    from two_track_h3 import mask_to_latent

    out = latents[0]
    for level in range(1, len(latents)):
        weight = mask_to_latent(masks[level - 1], 4, 0.0).to(out.device)
        out = weight * latents[level] + (1.0 - weight) * upscale(out)
    return out


def nested_masks(shapes, boxes) -> list[torch.Tensor]:
    """One box per level step, each intersected with the previous level's region (nesting)."""
    masks, allowed = [], None
    for level, box in enumerate(boxes):
        time_, height, width = shapes[level]
        x0, y0, x1, y1 = box
        mask = torch.zeros(time_, height, width, dtype=torch.bool)
        mask[:, round(y0 * height):round(y1 * height), round(x0 * width):round(x1 * width)] = True
        if allowed is not None:
            mask &= allowed
        masks.append(mask)
        allowed = mask.repeat_interleave(2, 1).repeat_interleave(2, 2)
    return masks


@torch.no_grad()
def pyramid_sample(dit, text, upscale, *, width, height, levels, steps, switches, boxes, seed, maps=None):
    """Draft sampler (module docstring). ``switches[l - 1]`` is the step where level ``l`` starts."""
    from h3_musubi import sigma_schedule

    sigmas = sigma_schedule(steps)
    full_h, full_w = height // TOKEN_PX, width // TOKEN_PX
    base = (1, full_h >> (levels - 1), full_w >> (levels - 1))
    shapes = [(1, base[1] << level, base[2] << level) for level in range(levels)]
    masks = nested_masks(shapes, boxes)
    gen = torch.Generator("cuda").manual_seed(seed)
    latents = [torch.randn(1, 24, 1, 2 * base[1], 2 * base[2], device="cuda", generator=gen)]
    active = 1
    tokens = []
    for index in range(steps):
        while active < levels and index == switches[active - 1]:
            # Their switch: upscale the clean estimate (here the composite of all active levels) and
            # re-noise it at the switch sigma. TODO(untested on GPU).
            sigma = float(sigmas[index])
            clean = upscale(compose(x0, masks, upscale))
            noise = torch.randn(clean.shape, device="cuda", generator=gen)
            latents.append((1.0 - sigma) * clean + sigma * noise)
            active += 1
        plan = PyramidPlan(base, masks[:active - 1]).to("cuda")
        sigma, sigma_next = float(sigmas[index]), float(sigmas[index + 1])
        velocity = pyramid_forward(dit, latents, plan, 1.0 - sigma, text, maps=maps)
        x0 = [lat + sigma * v.float() for lat, v in zip(latents, velocity)]
        ratio = sigma_next / sigma
        latents = [x + (1.0 - ratio) * sigma * v.float() for x, v in zip(latents, velocity)]
        tokens.append(sum(plan.counts))
    return compose(latents, masks, upscale), {"tokens": tokens, "plan": plan}


def plan_report(width: int, height: int, prefix: int, area: list[float] | None) -> dict:
    """CPU: how many levels fit, and tokens / attention pairs for a centred nested-box layout."""
    token_h, token_w = height // TOKEN_PX, width // TOKEN_PX
    report = {"canvas": f"{width}x{height}", "full_token_grid": [token_h, token_w],
              "max_levels": max_levels(token_h, token_w)}
    rows = []
    for levels in range(2, report["max_levels"] + 1):
        shares = area or ([1.0, 0.25] if levels == 2 else [1.0, 0.5, 0.15] if levels == 3 else
                          [1.0] + [0.5 ** level for level in range(1, levels)])
        base = (1, token_h >> (levels - 1), token_w >> (levels - 1))
        shapes = [(1, base[1] << level, base[2] << level) for level in range(levels)]
        # Centred boxes whose areas match the shares (approximately, on whole tokens).
        boxes = []
        for share in shares[1:]:
            side = share ** 0.5
            boxes.append((0.5 - side / 2, 0.5 - side / 2, 0.5 + side / 2, 0.5 + side / 2))
        plan = PyramidPlan(base, nested_masks(shapes, boxes))
        cost = attention_cost(plan, prefix)
        rows.append({"levels": levels, "coarsest_px": [base[2] * TOKEN_PX * 1, base[1] * TOKEN_PX],
                     "area_shares": shares, "token_share_formula": round(token_share(shares), 4),
                     "tokens": cost["tokens"], "full_tokens": plan.grid_counts[-1],
                     "token_share": round(cost["token_share"], 4), "attention_pair_share": round(cost["pair_share"], 4),
                     "finest_query_keys": cost["finest_queries_keys"]})
    report["layouts"] = rows
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("phase", choices=("plan", "render"))
    parser.add_argument("--size", type=int, nargs=2, default=(2688, 1536), metavar=("W", "H"))
    parser.add_argument("--prefix", type=int, default=300, help="plan: text + audio rows (attention cost only)")
    parser.add_argument("--area", type=float, nargs="+", default=None, help="plan: area share per level, level 0 = 1")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--switches", type=int, nargs="+", default=[14, 17])
    parser.add_argument("--boxes", type=float, nargs="+", default=[0.2, 0.1, 0.8, 0.95, 0.35, 0.15, 0.65, 0.5],
                        help="render: one x0 y0 x1 y1 box per level step, nested")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--swap", default="auto")
    args = parser.parse_args()
    width, height = args.size
    if args.phase == "plan":
        print(json.dumps(plan_report(width, height, args.prefix, args.area), indent=1))
        return
    # render: GPU, untested draft.
    from gpu_guard import auto_swap, refuse_if_busy
    from render_h3 import to_image
    from two_track_h3 import held_prompts, load_base, load_upscaler

    refuse_if_busy("pyramid_h3")
    report = plan_report(width, height, 0, None)
    if args.levels > report["max_levels"]:
        raise SystemExit(f"{width}x{height} allows {report['max_levels']} levels (token grid {report['full_token_grid']})")
    boxes = [tuple(args.boxes[i:i + 4]) for i in range(0, len(args.boxes), 4)][:args.levels - 1]
    dit = load_base(auto_swap(args.swap))
    upscale = load_upscaler()
    args.out.mkdir(parents=True, exist_ok=True)
    prompts = held_prompts(argparse.Namespace(cache=None, holdout=24, prompts=1))
    for index, (_caption, embeds) in enumerate(prompts):
        start = time.perf_counter()
        latent, info = pyramid_sample(dit, embeds.to("cuda", torch.bfloat16), upscale, width=width, height=height,
                                      levels=args.levels, steps=args.steps, switches=args.switches, boxes=boxes,
                                      seed=args.seed + index)
        to_image(upscale.vae.decode(latent.to("cuda", upscale.dtype))).save(args.out / f"p{index}_pyramid.png")
        print(f"LOT_H3 kind=pyramid_render prompt={index} seconds={time.perf_counter() - start:.1f} "
              f"mean_tokens={sum(info['tokens']) / len(info['tokens']):.0f}", flush=True)


if __name__ == "__main__":
    main()
