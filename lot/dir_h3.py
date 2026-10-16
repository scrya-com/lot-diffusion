"""Data-space iteration (DiR) for H3, with optional Level-of-Token layouts per iteration.

Paper: arXiv 2610.12102, "Few-Step Generation via Data-Space Iteration". A few-step
generator refines its own clean estimate instead of walking the noise schedule:

    x0_hat[0] = 0;   x0_hat[r+1] = G(x0_hat[r], z, r),   r = 0 .. R-1

with the SAME noise ``z`` at every iteration and no renoising. ``G`` is the H3 DiT,
changed minimally:

* ``z`` goes through the original ``video_patch_proj`` (or the LoT extent heads);
* the previous estimate goes through a NEW zero-initialised linear (``DirHead.x0_proj``)
  whose output is added to the embedded video rows, so an untrained ``G`` equals the
  one-step base prediction up to bf16 rounding (the extra temb row; bounded in ``parity_dir_h3.py``);
* an embedding of ``r`` (``DirHead.r_embed[r]``) is added to the video's timestep
  embedding, as a separate temb row used by the video rows only (see
  ``h3_musubi.video_modulation_rewrite``);
* the flow time is pinned to the noise end: H3 ``model_t_video = 0`` (sigma = 1).

Optional sigma schedule (``parse_sigmas``; default all 1, the paper's DiR). With ``sigmas``
below 1, iteration ``r`` sees the deterministic Euler state ``x_r`` instead of ``z`` at flow time
``sigma_r``, ``x0_hat[r+1] = x_r + sigma_r * out``, and ``x_{r+1} = (s_{r+1}/s_r) x_r +
(1 - s_{r+1}/s_r) x0_hat[r+1]``, still from ONE ``z`` and no renoising. All-ones keeps ``x_r = z``
(identical to the pinned form). A grid primes the iterations for any init: with a zero head the
generator IS its LoRA's R-step Euler sampler at step 0 (base H3's with no LoRA, ``grid``, shift 12;
Minimax-H3-Turbo's with the Turbo init, ``turbo``, shift 6), so later iterations refine from the
start rather than having to learn to read ``x0_proj`` (which starts at zero and moves at most ~lr
per Adam step).

H3 head convention (this repo): the head predicts ``x0 - eps`` and the clean estimate is
``x_t + sigma * out``. At sigma = 1, ``x_t = z``, so ``x0_hat = z + out``. With a LoT
layout the output is y-space velocity and ``x0_hat = splice.to_x(z + out_y)``.

Width of the timestep embedding. ``temb_dim`` is whatever ``dit._timestep_embeddings``
returns for this checkpoint, not a constant: 8 for the table-pruned checkpoint,
``adaln_rank + 1`` for the self-pruned one, ``time_embed_dim`` for an unpruned model.
``probe_temb_dim`` reads it from a call.

The DMD fake model reuses the same trick with its own ``fake_r_embed`` (``DirFakeHead``),
kept in a separate module so generator and fake parameters never share an optimizer.
"""

from __future__ import annotations

import contextlib

import torch
from torch import nn

from h3 import GRID_LAYOUTS, grid_layout, patchify
from h3_musubi import forward_video
from h3_splice import LotSplice
from layout import LotLayout

TOKEN_DIM = 96
DEFAULT_R = 4


def probe_temb_dim(dit) -> int:
    """Width of one timestep-embedding row for ``dit`` (checkpoint dependent)."""
    device = dit.video_patch_proj.weight.device
    with torch.no_grad():
        return int(dit._timestep_embeddings(torch.zeros(1), device).shape[-1])


class DirHead(nn.Module):
    """The generator's new parameters: previous-estimate projection and the iteration embedding."""

    def __init__(self, hidden: int, temb_dim: int, iterations: int = DEFAULT_R):
        super().__init__()
        self.iterations = int(iterations)
        self.x0_proj = nn.Linear(TOKEN_DIM, hidden)
        nn.init.zeros_(self.x0_proj.weight)
        nn.init.zeros_(self.x0_proj.bias)
        self.r_embed = nn.Parameter(torch.zeros(self.iterations, temb_dim))


class DirFakeHead(nn.Module):
    """The fake model's iteration embedding, apart from the generator's parameters."""

    def __init__(self, temb_dim: int, iterations: int = DEFAULT_R):
        super().__init__()
        self.iterations = int(iterations)
        self.fake_r_embed = nn.Parameter(torch.zeros(self.iterations, temb_dim))


# ------------------------------------------------------------------------------ pooling

_POOL_CACHE: dict = {}


def _pool_index(layout: LotLayout):
    """``(src, dst, counts)`` for ``layout``: dense raster index of every covered token,
    the rect it belongs to (in ``layout.rects`` order), and the tokens per rect. ``None``
    when pooling is the identity (a raster-ordered all-1x1 layout)."""
    cached = _POOL_CACHE.get(id(layout))
    if cached is not None and cached[0] is layout:
        return cached[1]
    height, width = layout.height, layout.width
    src, dst, counts = [], [], []
    identity = True
    for index, rect in enumerate(layout.rects):
        covered = [(rect.t + dt) * height * width + (rect.u + du) * width + (rect.v + dv)
                   for dt in range(rect.et) for du in range(rect.eh) for dv in range(rect.ew)]
        if len(covered) != 1 or covered[0] != index:
            identity = False
        src.extend(covered)
        dst.extend([index] * len(covered))
        counts.append(len(covered))
    result = None if identity else (torch.tensor(src), torch.tensor(dst), torch.tensor(counts, dtype=torch.float32))
    _POOL_CACHE[id(layout)] = (layout, result)
    return result


def pool_rows(rows: torch.Tensor, layout: LotLayout) -> torch.Tensor:
    """Mean of each rect's 1x1 tokens: ``(1, n_dense, D)`` to ``(1, layout.count, D)``.

    ``rows`` are in dense patchify order (T, H, W row-major, the order of ``pack_video_rows``);
    the output follows ``layout.rects`` order, the order ``LotSplice.embed_rows`` uses. A
    dense raster layout is the identity (the input is returned as is).
    """
    if rows.ndim != 3 or rows.shape[1] != layout.dense_count:
        raise ValueError(f"rows {tuple(rows.shape)} do not cover the layout's {layout.dense_count} dense tokens")
    index = _pool_index(layout)
    if index is None:
        return rows
    src, dst, counts = (t.to(rows.device) for t in index)
    pooled = rows.new_zeros(rows.shape[0], layout.count, rows.shape[2])
    pooled.index_add_(1, dst, rows[:, src])
    return pooled / counts.to(rows.dtype).view(1, -1, 1)


# --------------------------------------------------------------------------- generator

def estimate_extra(head: DirHead, estimate: torch.Tensor, layout: LotLayout | None) -> torch.Tensor:
    """``(1, count, hidden)`` embedding of the previous estimate, pooled to the layout's rects.

    ``estimate`` is ``(1, 24, T, H, W)`` in the DiT's input space (y-space for LoT). The
    projection is linear, so pooling the 96-wide tokens first equals projecting then
    pooling, at a fraction of the memory.
    """
    tokens = patchify(estimate.float()).flatten(1, 3)
    if layout is not None:
        tokens = pool_rows(tokens, layout)
    return head.x0_proj(tokens)


def parse_sigmas(spec, iters: int) -> list[float]:
    """Per-iteration flow sigma: ``1`` (the paper, pinned), ``grid`` / ``grid:SHIFT`` (H3's
    deterministic Euler grid ``shift(linspace(1, 0, R + 1)[:R])``, default shift 12 = H3's own,
    for any init: base H3, a DiR run, a fresh LoRA), ``turbo`` / ``turbo:SHIFT`` (the same grid
    at Turbo's 768p shift 6), or ``R`` comma-separated values in (0, 1] starting at 1."""
    if spec is None or str(spec) in ("1", "1.0", "pinned"):
        return [1.0] * iters
    if isinstance(spec, (list, tuple)):
        values = [float(v) for v in spec]
    elif str(spec).split(":", 1)[0] in ("grid", "turbo"):
        from h3_musubi import VIDEO_SHIFT, shift_sigma

        name, _, value = str(spec).partition(":")
        shift = float(value) if value else (6.0 if name == "turbo" else float(VIDEO_SHIFT))
        grid = torch.linspace(1.0, 0.0, iters + 1, dtype=torch.float64)[:iters]
        values = [float(shift_sigma(q, shift)) for q in grid]
    else:
        values = [float(v) for v in str(spec).split(",")]
    if len(values) != iters or values[0] != 1.0 or any(not 0.0 < v <= 1.0 for v in values) \
            or any(b > a for a, b in zip(values, values[1:])):
        raise ValueError(f"sigmas {spec!r}: need {iters} non-increasing values in (0, 1] starting at 1")
    return values


def next_state(state: torch.Tensor, clean: torch.Tensor, sigma: float, sigma_next: float) -> torch.Tensor:
    """Deterministic Euler step of the iteration state (``state`` unchanged when both sigmas are 1)."""
    if sigma_next == sigma:
        return state
    ratio = sigma_next / sigma
    return ratio * state + (1.0 - ratio) * clean.to(state.dtype)


def generator_step(dit, head: DirHead, z: torch.Tensor, estimate: torch.Tensor, r: int, text: torch.Tensor, *,
                   splice: LotSplice | None = None, sigma: float = 1.0,
                   audio_generator: torch.Generator | None = None) -> torch.Tensor:
    """One iteration ``x0_hat[r+1] = G(x0_hat[r], z, r)``, in x-space.

    Dense: ``z + sigma * out`` (``z`` is the iteration state ``x_r``; plain ``z`` when sigma = 1).
    LoT (sigma 1 only): ``splice.to_x(z + out_y)``, with ``z`` and ``estimate`` mapped into
    y-space for the DiT (``estimate`` is mapped by ``splice.to_y``).
    """
    state, previous = z, estimate
    if splice is not None:
        if sigma != 1.0:
            raise ValueError("a sigma schedule below 1 is dense-only")
        previous = splice.to_y(estimate)
    extra = estimate_extra(head, previous, None if splice is None else splice.layout)
    out = forward_video(dit, state, 1.0 - sigma, text, lot=splice, video_extra=extra,
                        video_temb_delta=head.r_embed[r], audio_generator=audio_generator).float()
    clean = state.float() + sigma * out
    return clean if splice is None else splice.to_x(clean)


# --------------------------------------------------------------------- DMD predictions

def fake_x0(dit, fake_head: DirFakeHead, x_tau: torch.Tensor, tau, r: int, text: torch.Tensor,
            audio_generator: torch.Generator | None = None) -> torch.Tensor:
    """The fake model's clean estimate at noise level ``tau`` (sigma), conditioned on ``r``."""
    out = forward_video(dit, x_tau, 1.0 - float(tau), text, video_temb_delta=fake_head.fake_r_embed[r],
                         audio_generator=audio_generator).float()
    return x_tau.float() + float(tau) * out


def fake_velocity(dit, fake_head: DirFakeHead, x_tau: torch.Tensor, tau, r: int, text: torch.Tensor,
                  audio_generator: torch.Generator | None = None) -> torch.Tensor:
    """The fake model's raw head output (``x0 - eps``), for its regression loss."""
    return forward_video(dit, x_tau, 1.0 - float(tau), text, video_temb_delta=fake_head.fake_r_embed[r],
                         audio_generator=audio_generator).float()


def real_x0(dit, x_tau: torch.Tensor, tau, text: torch.Tensor,
            audio_generator: torch.Generator | None = None) -> torch.Tensor:
    """The frozen base model's clean estimate at ``tau`` (call with every LoRA off)."""
    out = forward_video(dit, x_tau, 1.0 - float(tau), text, audio_generator=audio_generator).float()
    return x_tau.float() + float(tau) * out


@contextlib.contextmanager
def use_networks(gen_network, fake_network, *, gen: bool, fake: bool):
    """Switch the generator and fake LoRAs fully on or off (``.enabled``: no graph, no compute), then restore."""
    saved = []
    for network, on in ((gen_network, gen), (fake_network, fake)):
        if network is None:
            continue
        for module in network.unet_loras:
            saved.append((module, module.enabled))
            module.enabled = bool(on)
    try:
        yield
    finally:
        for module, previous in saved:
            module.enabled = previous


# ------------------------------------------------------------------------- schedule

def parse_layout_schedule(spec: str, iterations: int, token_h: int, token_w: int, latent_t: int) -> list[LotLayout | None]:
    """``"bands,bands,dense,dense"`` to one layout per iteration; ``None`` means dense.

    A single name repeats for every iteration. Names are ``h3.GRID_LAYOUTS`` (``dense``,
    ``bands``, ``center``, ``uniform2``, ``uniform4``), built by ``h3.grid_layout`` for any
    token grid. Equal names share one layout object.
    """
    names = [name.strip() for name in spec.split(",") if name.strip()]
    if len(names) == 1:
        names = names * iterations
    if len(names) != iterations:
        raise ValueError(f"schedule {spec!r} has {len(names)} entries, expected 1 or {iterations}")
    built: dict[str, LotLayout | None] = {}
    for name in names:
        if name not in GRID_LAYOUTS:
            raise ValueError(f"layout {name!r} is not one of {GRID_LAYOUTS}")
        if name not in built:
            built[name] = None if name == "dense" else grid_layout(name, latent_t, token_h, token_w)
    return [built[name] for name in names]


# ------------------------------------------------------------------------- sampling

@torch.no_grad()
def dir_sample(dit, head: DirHead, text: torch.Tensor, *, height: int, width: int, latent_t: int = 1,
               iters: int = DEFAULT_R, seed: int = 0, schedule: list | None = None, adapter=None,
               sigmas: list | None = None, device="cuda") -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run ``iters`` iterations from the same noise. Returns the final x-space latent and every ``x0_hat``.

    ``schedule`` is the list from ``parse_layout_schedule`` (``None`` or entries ``None``: dense).
    A non-dense entry needs ``adapter`` (a ``LotVisualAdapter``). ``sigmas`` from ``parse_sigmas``
    (default: all 1).
    """
    sigmas = sigmas or [1.0] * iters
    generator = torch.Generator(device).manual_seed(seed)
    z = torch.randn(1, 24, latent_t, height // 16, width // 16, device=device, generator=generator)
    estimate = torch.zeros_like(z)
    state = z
    schedule = schedule or [None] * iters
    if len(schedule) < iters:
        raise ValueError(f"schedule has {len(schedule)} entries for {iters} iterations")
    estimates = []
    for r in range(iters):
        layout = schedule[r]
        splice = None
        if layout is not None:
            if adapter is None:
                raise ValueError("a non-dense layout in the schedule needs an adapter")
            splice = LotSplice(adapter, layout, y_space=True)
        # same silent-audio noise at every iteration (a fresh generator with one seed), like z
        audio_generator = torch.Generator(device).manual_seed(seed + 1)
        estimate = generator_step(dit, head, state, estimate, r, text, splice=splice, sigma=sigmas[r],
                                  audio_generator=audio_generator)
        estimates.append(estimate)
        if r + 1 < iters:
            state = next_state(state, estimate, sigmas[r], sigmas[r + 1])
    return estimate, estimates


# ------------------------------------------------------------------------ run loading

def save_dir_run(run_dir, tag: str, *, head: DirHead, fake_head, gen_network, fake_network, adapter, meta: dict,
                 gen_only: bool = False) -> None:
    """Write one checkpoint of a ``train_dir_h3.py`` run: ``dir_head{tag}.pt``, ``dir_fake_head{tag}.pt``,
    ``lora_gen{tag}.safetensors``, ``lora_fake{tag}.safetensors``, ``adapter{tag}.pt`` (LoT runs) and ``config.json``.
    ``gen_only`` skips the fake model's files (used for the ``*_ema`` copies, tag ``"{tag}_ema"``)."""
    import json
    from pathlib import Path

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), run_dir / f"dir_head{tag}.pt")
    if not gen_only:
        torch.save(fake_head.state_dict(), run_dir / f"dir_fake_head{tag}.pt")
    gen_network.save_weights(str(run_dir / f"lora_gen{tag}.safetensors"), torch.bfloat16, {"lot_h3": "dir", "kind": "gen", "rank": str(meta["rank"])})
    if not gen_only:
        fake_network.save_weights(str(run_dir / f"lora_fake{tag}.safetensors"), torch.bfloat16, {"lot_h3": "dir", "kind": "fake", "rank": str(meta.get("fake_rank", meta["rank"]))})
    if adapter is not None:
        torch.save(dict(adapter.state_dict()), run_dir / f"adapter{tag}.pt")
    (run_dir / "config.json").write_text(json.dumps(meta, indent=1))


def load_dir_run(run_dir, dit, *, tag: str = "", device="cuda", adapter=None, ema: bool = False):
    """Load a generator saved by ``train_dir_h3.py`` onto ``dit``.

    Applies the generator LoRA to ``dit`` (rank read from the file) and returns
    ``(head, network, meta)``: the ``DirHead`` on ``device`` (eval mode), the LoRA network (leave
    it enabled to sample; ``use_networks`` / ``set_lora`` switch it off for the base model), and the
    ``config.json`` dict (``iters``, ``schedule``, ``rank``, ``temb_dim``, ``hidden``, ...).
    ``tag`` picks an intermediate save (``"_3"`` for ``--save-every`` step 3; ``""`` is the final one).
    ``ema=True`` loads the ``*_ema`` copy (``--ema`` runs; ``tag + "_ema"``).
    If ``adapter`` (a ``LotVisualAdapter``) is given and the run saved ``adapter{tag}.pt``, its state is
    loaded into it; the caller still builds the adapter and fits the bank as ``train_dir_h3.py`` does.
    """
    import json
    from pathlib import Path

    from safetensors import safe_open

    from h3_musubi import make_lora

    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "config.json").read_text())
    if ema:
        tag = f"{tag}_ema"
    lora_file = run_dir / f"lora_gen{tag}.safetensors"
    with safe_open(str(lora_file), "pt") as handle:
        key = next(k for k in handle.keys() if k.endswith("lora_down.weight"))
        rank = int(handle.get_slice(key).get_shape()[0])
    network = make_lora(dit, rank)
    info = network.load_weights(str(lora_file))
    if info.missing_keys:
        raise RuntimeError(f"{lora_file}: missing {len(info.missing_keys)} LoRA keys")
    network.requires_grad_(False)
    network.to(device=device, dtype=torch.bfloat16)
    head = DirHead(int(meta["hidden"]), int(meta["temb_dim"]), int(meta["iters"]))
    head.load_state_dict(torch.load(run_dir / f"dir_head{tag}.pt", map_location="cpu", weights_only=True))
    head.to(device).eval().requires_grad_(False)
    if adapter is not None and (run_dir / f"adapter{tag}.pt").is_file():
        adapter.load_state_dict(torch.load(run_dir / f"adapter{tag}.pt", map_location="cpu", weights_only=True))
    return head, network, meta
