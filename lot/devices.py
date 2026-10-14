"""Device helpers so the H3 scripts run on CUDA, Apple Silicon (MPS) or the CPU.

On CUDA every helper is the call the scripts made before (``torch.cuda.synchronize``,
``max_memory_allocated``, ...), so CUDA runs are unchanged. MPS has no float64, no
peak-memory counter, no CUDA streams (block swap) and no int8 kernels; see
``rotary_positions``, ``peak_mb``, ``swap_for`` and ``h3_musubi.load_dit(dequant_bf16=...)``.
"""

from __future__ import annotations

import torch

DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")

# None: narrow float64 rotary positions to float32 on the host only for MPS. True forces it on any
# device (CPU tests of the MPS path), False never.
FLOAT32_POSITIONS: bool | None = None


def mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def pick_device(name: str = "auto") -> torch.device:
    """``auto`` is cuda if available, else mps, else cpu; any other name is taken as given."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("mps" if mps_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda: CUDA is not available")
    if name == "mps" and not mps_available():
        raise SystemExit("--device mps: MPS is not available (Apple Silicon, macOS 14+, PyTorch with MPS)")
    return torch.device(name)


def add_device_arg(parser) -> None:
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto",
                        help="auto = cuda if available, else mps, else cpu")


def synchronize(device) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def reset_peak(device) -> None:
    """Reset the peak counter (CUDA only; MPS keeps none)."""
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_mb(device) -> float:
    """CUDA: peak allocated MiB since ``reset_peak``. MPS: the driver's current allocation (no peak
    counter exists). CPU: 0."""
    device = torch.device(device)
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 2**20
    return 0.0


def allocated_mb(device) -> float:
    device = torch.device(device)
    if device.type == "cuda":
        return torch.cuda.memory_allocated() / 2**20
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 2**20
    return 0.0


def empty_cache(device) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def swap_for(device, swap: int) -> int:
    """Block swap streams weights over CUDA streams from pinned memory; elsewhere it is off (with a note).
    On MPS the memory is unified: there is nothing to stream to."""
    if swap and torch.device(device).type != "cuda":
        print(f"LOT_H3 note: --swap {swap} needs CUDA; running with swap 0 on {torch.device(device).type}", flush=True)
        return 0
    return swap


def narrow_positions(device) -> bool:
    """Whether float64 rotary positions go to ``device`` as float32 (MPS, or ``FLOAT32_POSITIONS``)."""
    return torch.device(device).type == "mps" if FLOAT32_POSITIONS is None else FLOAT32_POSITIONS


def rotary_positions(positions: torch.Tensor, device) -> torch.Tensor:
    """``positions`` (float64, CPU) on ``device`` for ``_rotation_table``. MPS has no float64, so they are
    narrowed to float32 on the host first; ``_rotation_table`` narrows to float32 anyway, so the table is
    the same. CUDA moves the float64 tensor as before."""
    if narrow_positions(device) and positions.dtype == torch.float64:
        positions = positions.to(torch.float32)
    return positions.to(device)
