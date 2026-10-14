#!/usr/bin/env python3
"""DiR plumbing parity on one small latent: the generalised forward against the dense H3 forward.

Checks, on the pruned int8 checkpoint with block streaming (writes no file, refuses while another
LoT GPU script is alive, does not call empty_cache):

(a) ``forward_video(lot=None)`` equals ``predict(lot=None)`` (the stock ``dit(...)`` call).
(b) a zero ``video_temb_delta`` and zero ``video_extra`` still equal it.
(c) a nonzero delta changes the video output, and the rewritten modulation indices leave every
    non-video row (text, audio) untouched.
(d) an untrained ``DirHead`` generator step at r=0 equals ``z + forward_video(z, t=0, delta=0)`` (same
    thresholds), and is within ``T0_BOUND`` of the stock ``z + predict(z, t=0)``.

Thresholds are those of ``parity_h3.py``: rel_rms <= max(1e-4, 2 x run-to-run floor), max_abs <= 1e-2.

Why (d) has a looser second bound. At t=0 the audio shares the video's timestep (audio sigma 1 too),
so the stock forward has ONE timestep row and the AdaLN projection is a 1-row bf16 matmul. Appending the
delta row makes it a 2-row matmul, which rounds differently in the last bf16 bit (measured: block 5
adaln_proj, max |diff| 4e-6; block 20, 1e-4), and the 50-block stack amplifies that to a few percent.
(a)-(c) are exact because there the row count does not change kernel (at t = 0.6 or 0.01 the stock
forward already has 2 rows). ``T0_BOUND`` is a noise ceiling for that one case, not a tolerance on a bug:
a wrong row or index changes the output by O(1) (see c_moved).

    python3 lot/parity_dir_h3.py [--device mps] [--dequant]

``--dequant`` runs it on the bf16-dequantized base (``load_dit(dequant_bf16=True)``, the MPS default).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lot_paths import LOT_H3_CHECKPOINT  # noqa: E402

from devices import add_device_arg, peak_mb, pick_device, swap_for  # noqa: E402
from gpu_guard import refuse_if_busy  # noqa: E402

CHECKPOINT = LOT_H3_CHECKPOINT
T0_BOUND = 0.15


def rel_rms(left: torch.Tensor, right: torch.Tensor) -> float:
    delta = (left.float() - right.float()).pow(2).mean().sqrt()
    scale = right.float().pow(2).mean().sqrt().clamp(min=1e-8)
    return float(delta / scale)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_device_arg(parser)
    parser.add_argument("--dequant", action="store_true", help="bf16-dequantized base (default on non-CUDA devices)")
    args = parser.parse_args()
    refuse_if_busy("parity_dir_h3.py")
    device = pick_device(args.device)
    if not CHECKPOINT.is_file():
        raise SystemExit(f"missing checkpoint {CHECKPOINT}")
    from dir_h3 import DirHead, generator_step, probe_temb_dim
    from h3_musubi import (_layout, audio_sigma, forward_video, load_dit, predict, video_modulation_rewrite)
    from musubi_tuner.minimax_h3.packing import build_timestep_rows

    model = load_dit(CHECKPOINT, device=device, swap=swap_for(device, 32), dequant_bf16=args.dequant or None)
    torch.manual_seed(0)
    height = width = 16
    z = torch.randn(1, 24, 1, height, width, device=device, dtype=torch.float32)
    text = torch.randn(1, 4, model.config.text_dim, device=device, dtype=torch.bfloat16)
    temb_dim = probe_temb_dim(model)
    hidden = model.config.hidden_size
    count = (height // 2) * (width // 2)

    def audio():
        return torch.Generator(device).manual_seed(0)

    def stock(t):
        with torch.inference_mode():
            return predict(model, z, t, text, audio_generator=audio())

    def hooked(t, **kw):
        with torch.inference_mode():
            return forward_video(model, z, t, text, audio_generator=audio(), **kw)

    t = 0.6
    base = stock(t)
    floor = rel_rms(stock(t), base)
    limit = max(1e-4, 2.0 * floor)

    def within(out, ref):
        return rel_rms(out, ref) <= limit and float((out.float() - ref.float()).abs().max()) <= 1e-2

    gap_a = rel_rms(hooked(t), base)
    ok_a = within(hooked(t), base)
    zero_delta = torch.zeros(temb_dim, device=device)
    zero_extra = torch.zeros(1, count, hidden, device=device)
    gap_b = rel_rms(hooked(t, video_temb_delta=zero_delta, video_extra=zero_extra), base)
    ok_b = within(hooked(t, video_temb_delta=zero_delta, video_extra=zero_extra), base)

    delta = 0.5 * torch.randn(temb_dim, device=device)
    moved = rel_rms(hooked(t, video_temb_delta=delta), base)
    layout = _layout(text.shape[1], 1, height, width)
    rows = build_timestep_rows(layout, text_token_tags=torch.ones(1, 4, dtype=torch.int64),
                               model_t_video=t, model_t_audio=1.0 - float(audio_sigma(1.0 - t)))
    video_slice = layout.target_video_segment.row_slice
    new_index = len(rows.unique_timesteps)
    after, segments = video_modulation_rewrite(rows.block_adaln_indices[0], video_slice, new_index)
    others = torch.ones_like(after, dtype=torch.bool)
    others[video_slice] = False
    untouched = bool(torch.equal(after[others], rows.block_adaln_indices[0][others]))
    video_moved = bool(torch.all(after[video_slice] == 3 * new_index))
    ok_c = moved > 10 * limit and untouched and video_moved and segments[-1][2] == 3 * new_index

    head = DirHead(hidden, temb_dim).to(device)
    with torch.inference_mode():
        step = generator_step(model, head, z, torch.zeros_like(z), 0, text, audio_generator=audio())
    with torch.inference_mode():
        reference = z + forward_video(model, z, 0.0, text, video_temb_delta=zero_delta,
                                      video_extra=zero_extra, audio_generator=audio()).float()
    gap_d = rel_rms(step, reference)
    stock_gap = rel_rms(step, z + stock(0.0).float())
    ok_d = gap_d <= max(1e-4, 2.0 * floor) and float((step - reference).abs().max()) <= 1e-2 and stock_gap <= T0_BOUND

    ok = int(ok_a and ok_b and ok_c and ok_d and floor <= 1e-3)
    print(
        "LOT_H3 kind=dir_parity "
        f"temb_dim={temb_dim} hidden={hidden} rel_rms_floor={floor:.6g} "
        f"a={gap_a:.6g}/{int(ok_a)} b={gap_b:.6g}/{int(ok_b)} c_moved={moved:.6g}/{int(ok_c)} "
        f"d={gap_d:.6g}/{int(ok_d)} d_vs_stock_t0={stock_gap:.6g} peak_mb={peak_mb(device):.1f} dtype=bf16 ok={ok}"
    )
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
