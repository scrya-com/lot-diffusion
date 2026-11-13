"""One H3 GPU job at a time. The guard refuses; it never kills anything."""

from __future__ import annotations

import math
import os
import subprocess

GPU_SCRIPTS = ("train_synth.py", "parity_h3.py", "procrustes_h3.py", "make_pairs_h3.py", "render_h3.py", "time_decode_h3.py", "train_h3.py", "build_cache_h3.py", "time_lot_h3.py", "mixed_res_h3.py", "two_track_h3.py", "train_tt.py", "build_pairs_tt.py", "pyramid_h3.py")


def other_gpu_job() -> str | None:
    """Command line of another LoT GPU script on this machine, or ``None``."""
    listing = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True)
    me = os.getpid()
    for line in listing.stdout.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        argv = cmd.split()
        if not pid.isdigit() or int(pid) == me or not argv:
            continue
        # argv[0] must be the interpreter: a shell whose command string names a
        # script is a wrapper, not the job.
        if not os.path.basename(argv[0]).startswith("python"):
            continue
        if any(os.path.basename(arg) in GPU_SCRIPTS for arg in argv[1:]):
            return cmd
    return None


def refuse_if_busy(name: str) -> None:
    job = other_gpu_job()
    if job is not None:
        raise SystemExit(f"{name}: another LoT GPU job is running ({job}). It will not be killed.")


def card_gb() -> float:
    """Free memory on the current CUDA device, GB (other processes count against it)."""
    import torch

    free, _total = torch.cuda.mem_get_info()
    return free / 2**30


def auto_swap(value: str, inference: bool = False, latent_t: int = 7) -> int:
    """``--swap auto``: DiT blocks to stream from CPU, from the card's free memory.

    Measured with ``train_tt.py lora --check`` (two-track LoRA, 1344x768 pairs): 8 blocks peak at
    19.6 GB on an RTX 5090; 24 blocks peak at 13.3 GB on an RTX PRO 4000 with 6 GB already taken.
    0 (all resident) on 64 GB+ free, e.g. an RTX PRO 6000 (96 GB), is not measured yet."""
    if value != "auto":
        return int(value)
    gb = card_gb()
    if inference:
        # Rendering (no LoRA grads, no checkpointing), measured 2026-11-13 on an RTX 5090 at 1344x768, 22 frames,
        # VAE decode included: peak 26.8 GB with every block resident, about 0.42 GB less per streamed block.
        # Streaming costs two-track the most (LR steps are short): 14.0 s at 0-2 streamed, 20.9 s at 8.
        # Longer clips need more (activations, VAE decode): 22 latent frames (73 pixel frames) ran out of memory
        # with none streamed and fit with 6. Assumed 0.35 GB per latent frame beyond 7.
        need = 26.8 + 0.35 * max(0, latent_t - 7) + 1.0
        return 0 if gb >= need else min(50, int(math.ceil((need - gb) / 0.42)))
    return 0 if gb >= 64 else 8 if gb >= 28 else 16 if gb >= 21 else 24
