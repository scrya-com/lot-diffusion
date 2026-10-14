#!/usr/bin/env python3
"""Streaming-DiR plumbing on the real checkpoint: parity with the still DiR step and stock FL2VA, and timing.

Checks (pruned int8 checkpoint, block streaming; writes no file, refuses while another LoT GPU
script is alive):

(a) T2VA window, untrained ``StreamHead``, every frame at r = 0, no memory: ``stream_forward`` equals
    ``dir_h3.generator_step`` with an untrained ``DirHead`` on the same ``(1, 24, T, H, W)`` noise. Both
    append exactly one timestep row, so this is held to ``parity_dir_h3``'s thresholds.
(b) FL2VA-first window (frame 0 done, the rest r = 0): ``stream_forward`` against the stock
    ``dit(...)`` FL2VA call on ``z``. The stream adds two timestep rows to the stock two, which flips
    bf16 roundings in the AdaLN matmul (see ``parity_dir_h3`` (d)); bounded by ``T0_BOUND``, reported.
(c) the done-row embedding and a full memory move the output (O(1) wiring, not rounding).
(d) seconds and peak memory of one window forward at ``--time-size`` with and without a full memory.

    python3 lot/parity_stream_h3.py [--time-size 48 48] [--capacity 16] [--pool 4]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lot_paths import LOT_H3_CHECKPOINT  # noqa: E402

from gpu_guard import refuse_if_busy  # noqa: E402

T0_BOUND = 0.15


def rel_rms(left: torch.Tensor, right: torch.Tensor) -> float:
    delta = (left.float() - right.float()).pow(2).mean().sqrt()
    scale = right.float().pow(2).mean().sqrt().clamp(min=1e-8)
    return float(delta / scale)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--time-size", type=int, nargs=2, default=[48, 48], metavar=("H", "W"),
                        help="latent size of the timed window (48 48 = 768x768 pixels)")
    parser.add_argument("--frames", type=int, default=7, help="window latent frames (5n+2)")
    parser.add_argument("--capacity", type=int, default=16, help="memory keyframes in the timed run")
    parser.add_argument("--pool", type=int, default=4)
    parser.add_argument("--swap", type=int, default=32)
    args = parser.parse_args()

    refuse_if_busy("parity_stream_h3.py")
    if not LOT_H3_CHECKPOINT.is_file():
        raise SystemExit(f"missing checkpoint {LOT_H3_CHECKPOINT}")
    from dir_h3 import DirHead, generator_step, probe_temb_dim
    from h3_musubi import _silence, audio_sigma, load_dit
    from stream_dir_h3 import (MemoryEntry, StreamHead, check_window, coarse_tokens, memory_rows, pooled_centres,
                               rows_per_window, stream_forward, stream_layout)

    check_window(args.frames)
    model = load_dit(LOT_H3_CHECKPOINT, swap=args.swap)
    temb_dim, hidden = probe_temb_dim(model), model.config.hidden_size
    iterations = 4
    torch.manual_seed(0)
    frames, height, width = args.frames, 16, 16
    z = torch.randn(1, 24, frames, height, width, device="cuda")
    text = torch.randn(1, 4, model.config.text_dim, device="cuda", dtype=torch.bfloat16)
    head = StreamHead(hidden, temb_dim, iterations).cuda()
    still = DirHead(hidden, temb_dim, iterations).cuda()

    def audio():
        return torch.Generator("cuda").manual_seed(0)

    def stream(r, **kw):
        with torch.inference_mode():
            return stream_forward(model, head, z, kw.pop("estimate", torch.zeros_like(z)), torch.as_tensor(r), text,
                                  audio_generator=audio(), **kw)

    # (a) T2VA, all r = 0: the still DiR step, generalised to T frames
    zeros = [0] * frames
    with torch.inference_mode():
        reference = generator_step(model, still, z, torch.zeros_like(z), 0, text, audio_generator=audio())
        floor = rel_rms(generator_step(model, still, z, torch.zeros_like(z), 0, text, audio_generator=audio()), reference)
    limit = max(1e-4, 2.0 * floor)
    got = stream(zeros)
    gap_a = rel_rms(got, reference)
    ok_a = gap_a <= limit and float((got - reference).abs().max()) <= 1e-2

    # (b) FL2VA first: frame 0 done (its estimate is the condition), against the stock call
    first = torch.randn(1, 24, 1, height, width, device="cuda")
    estimate = torch.zeros_like(z)
    estimate[:, :, :1] = first
    r_first = [iterations] + [0] * (frames - 1)
    layout = stream_layout(text.shape[1], frames, height, width, first=True, memory=0)
    sigma_a = float(audio_sigma(1.0))
    with torch.inference_mode():
        stock = model(video_latents=z, audio_latents=_silence(frames, sigma_a, "cuda", audio()), text_hidden_states=text,
                      text_token_tags=torch.ones(1, 4, dtype=torch.int64), layout=layout, model_t_video=0.0,
                      model_t_audio=1.0 - sigma_a, visual_condition_latents=[first]).video.float()
    got_b = stream(r_first, estimate=estimate, first=first)
    gap_b = rel_rms(got_b[:, :, 1:], (z + stock)[:, :, 1:])
    ok_b = gap_b <= T0_BOUND

    # (c) wiring: the done row and the memory move the output by O(1)
    with torch.no_grad():
        head.r_embed[iterations].normal_(0.0, 0.5)
    moved_done = rel_rms(stream(r_first, estimate=estimate, first=first)[:, :, 1:], got_b[:, :, 1:])
    with torch.no_grad():
        head.r_embed.zero_()
    entries = [MemoryEntry(coarse_tokens(torch.randn(1, 24, 1, height, width), args.pool),
                           pooled_centres(height, width, args.pool), -60.0 * (k + 1)) for k in range(4)]
    moved_memory = rel_rms(stream(r_first, estimate=estimate, first=first,
                                  memory=memory_rows(entries[::-1], 0.0))[:, :, 1:], got_b[:, :, 1:])
    ok_c = moved_done > 10 * limit and moved_memory > 10 * limit

    # (d) timing of one window at the training / render size
    th, tw = args.time_size
    zt = torch.randn(1, 24, frames, th, tw, device="cuda")
    ft = torch.randn(1, 24, 1, th, tw, device="cuda")
    et = torch.zeros_like(zt)
    et[:, :, :1] = ft
    timed = {}
    for capacity in (0, args.capacity):
        entries = [MemoryEntry(coarse_tokens(torch.randn(1, 24, 1, th, tw), args.pool), pooled_centres(th, tw, args.pool),
                               -30.0 * (capacity - k)) for k in range(capacity)]
        memory = memory_rows(entries, 0.0) if capacity else None
        torch.cuda.reset_peak_memory_stats()
        seconds = []
        for _ in range(3):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                stream_forward(model, head, zt, et, torch.tensor(r_first), text, first=ft, memory=memory,
                               audio_generator=audio())
            torch.cuda.synchronize()
            seconds.append(time.perf_counter() - start)
        timed[capacity] = (min(seconds[1:]), torch.cuda.max_memory_allocated() / 2**20,
                           rows_per_window((frames, th, tw), memory=capacity, pool=args.pool, text_length=text.shape[1]))

    ok = int(ok_a and ok_b and ok_c and floor <= 1e-3)
    print("LOT_H3 kind=stream_parity "
          f"frames={frames} floor={floor:.6g} a={gap_a:.6g}/{int(ok_a)} b_vs_stock_fl2va={gap_b:.6g}/{int(ok_b)} "
          f"c_done={moved_done:.4g} c_memory={moved_memory:.4g}/{int(ok_c)} ok={ok}", flush=True)
    for capacity, (seconds, peak, rows) in timed.items():
        print(f"LOT_H3 kind=stream_timing size={th}x{tw} frames={frames} memory={capacity} rows={rows} "
              f"seconds={seconds:.2f} peak_mb={peak:.0f} swap={args.swap}", flush=True)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
