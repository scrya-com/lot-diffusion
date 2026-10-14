"""Streaming data-space iteration (DiR) for H3: a sliding window that is refined, never re-noised.

Builds on ``dir_h3`` (``x0_hat[r+1] = G(x0_hat[r], z, r)``, same ``z``, no schedule). A stream keeps
one window of ``T = 5n + 2`` latent frames and three things per frame: its noise ``z``, its current
clean estimate, and its own refinement index ``r`` (``R`` = done). One forward advances every
active frame by one iteration; when the leading frames are done the window slides by 5 latent
frames and only the new tail starts from ``r = 0`` (estimate 0, fresh noise). Nothing that was
already refined is regenerated.

Why 5 frames, and why the slide is exact. H3's VAE encodes every 17-pixel-frame clip on its own
(``clip_length=17``, 5 tokens per clip, no causal state across clips; ``video_vae.py``), and its
first token sees one pixel frame. So window frames 5k are clip starts, each the latent of a single
pixel frame, which is exactly what an FL2VA ``first`` condition is (``encode_video_condition`` of one
frame). After a slide by 5, the old frame 5 becomes frame 0 and is fed as the ``first`` condition
(the checkpoint is FL2VA, so this is H3's own trained context path, no new parameters); old frame 6
becomes frame 1 unchanged; no decode / re-encode is needed in latent space. Each slide emits frames
0..4, one whole VAE clip (17 pixel frames).

Per-frame ``r``. A window mixes done frames (the condition frame, a carried frame) with frames at
``r = 0``, which one global iteration embedding cannot express. Video rows are frame-major, so each
frame gets its own timestep row ``temb[video] + r_embed[r_f]`` (``frame_modulation_rewrite``), in the
blocks and in the final layer. ``StreamHead.r_embed`` has ``R + 1`` rows; row ``R`` marks a done frame.
Done frames keep their estimate (the forward's output for them is discarded).

Long-range memory (MilliViD-style coarse tokens). Each clip start that leaves the window is kept
as a few coarse tokens: the 96-wide patch tokens average-pooled ``pool x pool`` (6 x 6 tokens for a
48 x 48 latent at pool 4). They enter the sequence as clean visual-condition rows in front of the
``first`` condition, embedded by the base ``video_patch_proj`` plus a zero-initialised ``mem_proj``,
at the pooled block centres and at their real past times (``memory_time="real"``) or packed right
before the window (``"packed"``). The memory is bounded (``capacity`` keyframes), so the sequence
length stays bounded for any stream length.

Audio stays silent noise, as in ``dir_h3`` (the generator is trained without sound).

CPU tests: ``test_lot.py`` (``test_stream_*``). GPU parity: ``parity_stream_h3.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import torch
import torch.nn.functional as F
from torch import nn

from dir_h3 import DEFAULT_R, TOKEN_DIM, DirHead
from h3 import patchify

SLIDE = 5                 # latent frames per VAE clip (17 pixel frames)
CLIP_PIXEL_FRAMES = 17


def _packing():
    import h3_musubi  # noqa: F401  (puts MUSUBI_SRC on sys.path)
    from musubi_tuner.minimax_h3 import packing

    return packing


def clip_time() -> float:
    """Rotary time one VAE clip (5 latent frames, 17 pixel frames) spans."""
    return _packing().FRAME_RESCALE * CLIP_PIXEL_FRAMES


def check_window(frames: int) -> None:
    if frames < SLIDE + 2 or (frames - 2) % SLIDE:
        raise ValueError(f"a stream window needs 5n+2 latent frames with n >= 1, got {frames}")


# ------------------------------------------------------------------------------- head

class StreamHead(DirHead):
    """``DirHead`` with a done row in ``r_embed`` (``R + 1`` rows) and the memory projection.

    ``r_embed[R]`` (zero at init) is the timestep delta of done frames. ``mem_proj`` (zero at init) is
    added to the base ``video_patch_proj`` embedding of the coarse memory tokens.
    """

    def __init__(self, hidden: int, temb_dim: int, iterations: int = DEFAULT_R):
        super().__init__(hidden, temb_dim, iterations)
        self.r_embed = nn.Parameter(torch.zeros(self.iterations + 1, temb_dim))
        self.mem_proj = nn.Linear(TOKEN_DIM, hidden)
        nn.init.zeros_(self.mem_proj.weight)
        nn.init.zeros_(self.mem_proj.bias)

    def load_dir_head(self, state: dict) -> None:
        """Warm start from a still-image ``DirHead`` state (``dir_head*.pt`` of ``train_dir_h3.py``)."""
        if state["r_embed"].shape[0] != self.iterations:
            raise ValueError(f"DirHead has {state['r_embed'].shape[0]} iterations, the stream {self.iterations}")
        with torch.no_grad():
            self.x0_proj.weight.copy_(state["x0_proj.weight"])
            self.x0_proj.bias.copy_(state["x0_proj.bias"])
            self.r_embed[: self.iterations].copy_(state["r_embed"])


class StreamFakeHead(nn.Module):
    """The DMD fake model's per-frame iteration embedding (``R + 1`` rows, row ``R`` = done)."""

    def __init__(self, temb_dim: int, iterations: int = DEFAULT_R):
        super().__init__()
        self.iterations = int(iterations)
        self.fake_r_embed = nn.Parameter(torch.zeros(self.iterations + 1, temb_dim))

    def load_dir_fake_head(self, state: dict) -> None:
        """Warm start from a still run's ``dir_fake_head*.pt``."""
        with torch.no_grad():
            self.fake_r_embed[: self.iterations].copy_(state["fake_r_embed"])


def load_stream_run(run_dir, dit, *, tag: str = "", device="cuda", ema: bool = False):
    """``(StreamHead, gen LoRA network, config)`` of a ``train_stream_dir_h3.py`` run (``config["stream"]``),
    or a still ``train_dir_h3.py`` run warm-started into a ``StreamHead`` (done row and memory zero)."""
    import json
    from pathlib import Path

    from dir_h3 import load_dir_run

    config = json.loads((Path(run_dir) / "config.json").read_text())
    if not config.get("stream"):
        still, network, config = load_dir_run(run_dir, dit, tag=tag, device=device, ema=ema)
        head = StreamHead(int(config["hidden"]), int(config["temb_dim"]), still.iterations)
        head.load_dir_head({k: v.cpu() for k, v in still.state_dict().items()})
        return head.to(device).eval().requires_grad_(False), network, config
    from safetensors import safe_open

    from h3_musubi import make_lora

    sfx = f"{tag}_ema" if ema else tag
    lora_file = Path(run_dir) / f"lora_gen{sfx}.safetensors"
    with safe_open(str(lora_file), "pt") as handle:
        key = next(k for k in handle.keys() if k.endswith("lora_down.weight"))
        rank = int(handle.get_slice(key).get_shape()[0])
    network = make_lora(dit, rank)
    info = network.load_weights(str(lora_file))
    if info.missing_keys:
        raise RuntimeError(f"{lora_file}: missing {len(info.missing_keys)} LoRA keys")
    network.requires_grad_(False)
    network.to(device=device, dtype=torch.bfloat16)
    head = StreamHead(int(config["hidden"]), int(config["temb_dim"]), int(config["iters"]))
    head.load_state_dict(torch.load(Path(run_dir) / f"stream_head{sfx}.pt", map_location="cpu", weights_only=True))
    return head.to(device).eval().requires_grad_(False), network, config


# ----------------------------------------------------------------------------- memory

@dataclass
class MemoryEntry:
    tokens: torch.Tensor      # (n, 96) pooled patch tokens
    centres: torch.Tensor     # (n, 2) rotary (h, w) of the pooled blocks
    time: float               # absolute stream rotary time of the frame


def pooled_centres(height: int, width: int, pool: int) -> torch.Tensor:
    """``(n, 2)`` rotary (h, w) centres of ``pool x pool`` token blocks: the mean of the dense
    frame grid over each block (``pool = 1`` is the dense grid itself)."""
    packing = _packing()
    frame, _ = packing._frame_grid(packing.H3VideoGeometry(1, height, width))
    grid = frame.reshape(height // 2, width // 2, 2).permute(2, 0, 1).unsqueeze(0)
    pooled = F.avg_pool2d(grid, pool, ceil_mode=True)
    return pooled[0].permute(1, 2, 0).reshape(-1, 2)


def coarse_tokens(frame: torch.Tensor, pool: int) -> torch.Tensor:
    """``(n, 96)`` coarse tokens of one latent frame ``(1, 24, 1, H, W)``: patch tokens averaged
    over ``pool x pool`` blocks (edge blocks average what they cover)."""
    if frame.ndim != 5 or frame.shape[:3] != (1, 24, 1):
        raise ValueError(f"frame must be (1, 24, 1, H, W), got {tuple(frame.shape)}")
    tokens = patchify(frame.float())[0, 0]                       # (h, w, 96)
    pooled = F.avg_pool2d(tokens.permute(2, 0, 1).unsqueeze(0), pool, ceil_mode=True)
    return pooled[0].permute(1, 2, 0).reshape(-1, TOKEN_DIM)


def memory_rows(memory: list[MemoryEntry], window_time: float, mode: str = "real"):
    """``(tokens (M, 96), positions (M, 3))`` with times relative to the window's frame 0 (< 0).

    ``real``: each keyframe at its true stream time. ``packed``: keyframes one clip apart, ending one
    clip before the window, whatever their real age (bounded rotary distance for long streams).
    """
    if not memory:
        return None
    if mode not in ("real", "packed"):
        raise ValueError(f"memory_time {mode!r} is not real or packed")
    tokens, positions = [], []
    for index, entry in enumerate(memory):
        if mode == "real":
            relative = entry.time - window_time
        else:
            relative = -(len(memory) - index) * clip_time()
        if relative >= 0:
            raise ValueError("memory must lie before the window")
        tokens.append(entry.tokens)
        position = torch.empty(entry.tokens.shape[0], 3, dtype=torch.float64)
        position[:, 0] = relative
        position[:, 1:] = entry.centres
        positions.append(position)
    return torch.cat(tokens), torch.cat(positions)


# ----------------------------------------------------------------------------- layout

def stream_layout(text_length: int, frames: int, height: int, width: int, *, first: bool, memory: int):
    """The packed layout: ``text | memory | first | target_audio | target_video``.

    Without memory this is H3's own T2VA (``first=False``) or FL2VA-first layout. The memory block is
    one ``visual_condition`` segment (role ``memory``) of ``memory`` rows, so it gets the clean
    condition timestep like the ``first`` frame.
    """
    from h3_musubi import audio_frames

    packing = _packing()
    if first:
        layout = packing.build_h3_layout(task="fl2va", text_length=text_length, target_video=(frames, height, width),
                                         target_audio_frames=audio_frames(frames), visual_conditions=[(1, height, width)],
                                         condition_roles=("first",))
    else:
        layout = packing.build_h3_layout(task="t2va", text_length=text_length, target_video=(frames, height, width),
                                         target_audio_frames=audio_frames(frames))
    if not memory:
        return layout
    segments = []
    for segment in layout.segments:
        if segment.kind == "text":
            segments.append(segment)
            segments.append(packing.H3RowSegment("memory", "visual_condition", segment.stop, segment.stop + memory))
        else:
            segments.append(replace(segment, start=segment.start + memory, stop=segment.stop + memory))
    return replace(layout, segments=tuple(segments), row_count=layout.row_count + memory)


def stream_positions(layout, memory_positions: torch.Tensor | None = None) -> torch.Tensor:
    """``(rows, 3)`` rotary positions for ``stream_layout``.

    Text is ``0 .. L-1``. The window (``first``, target audio, target video) sits where H3 puts it,
    moved later by the age of the oldest memory row, so every relative time is kept; memory rows are
    at their relative time (``memory_positions[:, 0] < 0``) and pooled centres. With no memory this
    equals ``build_position_grid`` of the stock layout.
    """
    packing = _packing()
    target = layout.target_video
    frame, width_axis = packing._frame_grid(target)
    shift = 0.0 if memory_positions is None else float(-memory_positions[:, 0].min())
    window = float(layout.text_length) + shift
    positions = torch.empty(layout.row_count, 3, dtype=torch.float64)
    for segment in layout.segments:
        rows = segment.row_slice
        if segment.kind == "text":
            positions[rows] = 0
            positions[rows, 0] = torch.arange(layout.text_length, dtype=torch.float64)
        elif segment.role == "memory":
            if memory_positions is None or memory_positions.shape[0] != segment.row_count:
                raise ValueError("memory positions do not match the memory segment")
            positions[rows] = memory_positions.to(torch.float64)
            positions[rows, 0] += window
        elif segment.role == "first":
            positions[rows, 0] = window
            positions[rows, 1:] = frame
        elif segment.kind == "target_audio":
            positions[rows] = packing._audio_grid(window, layout.target_audio_frames,
                                                  float(width_axis[0]), float(width_axis[-1]))
        elif segment.kind == "target_video":
            positions[rows] = packing._video_grid(target, window)
        else:
            raise ValueError(f"unexpected segment {segment.role}")
    return positions


def frame_modulation_rewrite(block_adaln_indices: torch.Tensor, video_slice: slice, frame_rows: int,
                             frame_temb_index) -> tuple[torch.Tensor, tuple]:
    """Give every target-video frame its own timestep row (``video_modulation_rewrite`` per frame).

    ``frame_temb_index[f]`` is the temb row of frame ``f``; video rows are frame-major, ``frame_rows``
    per frame. Every other row keeps its index. Returns the new indices and the block segments.
    """
    packing = _packing()
    indices = block_adaln_indices.clone()
    count = video_slice.stop - video_slice.start
    frame_temb_index = torch.as_tensor(frame_temb_index, dtype=torch.int64)
    if count != frame_rows * len(frame_temb_index):
        raise ValueError(f"{count} video rows are not {len(frame_temb_index)} frames of {frame_rows}")
    indices[video_slice] = 3 * frame_temb_index.repeat_interleave(frame_rows)
    return indices, packing._build_modulation_segments(indices)


# ---------------------------------------------------------------------------- forward

def window_forward(dit, latent: torch.Tensor, t: float, text: torch.Tensor, *, first: torch.Tensor | None = None,
                   memory: tuple[torch.Tensor, torch.Tensor] | None = None, mem_proj: nn.Module | None = None,
                   frame_group: torch.Tensor | None = None, group_delta: torch.Tensor | None = None,
                   video_extra: torch.Tensor | None = None,
                   audio_generator: torch.Generator | None = None) -> torch.Tensor:
    """H3's raw video head output (``x0 - eps``) for a window ``latent`` ``(1, 24, T, H, W)`` at cleanness ``t``.

    The sequence is ``text | memory | first | audio | video`` (``stream_layout``). ``first`` is the
    ``(1, 24, 1, H, W)`` FL2VA condition, ``memory`` is ``memory_rows(...)`` (embedded by
    ``video_patch_proj`` plus ``mem_proj``). ``frame_group`` ``(T,)`` (an index per frame, e.g. its ``r``)
    with ``group_delta`` ``(G, temb_dim)`` gives each used group its own timestep row
    ``temb[video] + group_delta[g]``; grouping is by index, not by value, so two groups whose deltas are
    still equal (zero at init) get separate rows and separate gradients. ``video_extra``
    ``(1, T*h*w, hidden)`` is added to the embedded video rows. With none of them this is the stock
    T2VA / FL2VA forward (``test_stream_forward_matches_stock_h3``). The DMD real and fake models call
    it at ``t = 1 - tau``; the generator at ``t = 0`` (``stream_forward``).
    """
    from h3_musubi import _silence, _sync_swap, audio_sigma, run_blocks

    packing = _packing()
    if latent.ndim != 5 or latent.shape[0] != 1:
        raise ValueError(f"latent must be (1, 24, T, H, W), got {tuple(latent.shape)}")
    if text.ndim == 2:
        text = text.unsqueeze(0)
    _b, _c, frames, lat_h, lat_w = latent.shape
    _sync_swap(dit)
    device = dit.video_patch_proj.weight.device
    t_video = float(t)
    sigma_a = float(audio_sigma(1.0 - t_video))
    mem_count = 0 if memory is None else memory[0].shape[0]
    layout = stream_layout(text.shape[1], frames, lat_h, lat_w, first=first is not None, memory=mem_count)

    video = dit.video_patch_proj(packing.pack_video_rows(latent.to(device, torch.float32))).to(dit.dtype)
    if video_extra is not None:
        video = video + video_extra.to(device=video.device, dtype=video.dtype)
    parts = [dit.token_refiner(dit.condition_proj(text.to(device, dit.dtype)))]
    if mem_count:
        tokens = memory[0].to(device, torch.float32).unsqueeze(0)
        embedded = dit.video_patch_proj(tokens)
        if mem_proj is not None:
            embedded = embedded + mem_proj(tokens)
        parts.append(embedded.to(dit.dtype))
    if first is not None:
        parts.append(dit.video_patch_proj(packing.pack_video_rows(first.to(device, torch.float32))).to(dit.dtype))
    audio = _silence(frames, sigma_a, device, audio_generator)
    parts.append(dit.audio_patch_proj(packing.pack_audio_rows(audio).to(torch.float32)).to(dit.dtype))
    parts.append(video)
    hidden = torch.cat(parts, dim=1)
    if hidden.shape[1] != layout.row_count:
        raise RuntimeError(f"packed {hidden.shape[1]} rows for a {layout.row_count}-row layout")

    positions = stream_positions(layout, None if memory is None else memory[1])
    rotation = dit._rotation_table(positions.unsqueeze(0).to(device), hidden.dtype)
    tags = torch.ones(1, text.shape[1], dtype=torch.int64)
    rows = packing.build_timestep_rows(layout, text_token_tags=tags, model_t_video=t_video, model_t_audio=1.0 - sigma_a)
    temb = dit._timestep_embeddings(rows.unique_timesteps, device)
    frame_rows = (lat_h // 2) * (lat_w // 2)
    block_segments = rows.block_segments
    row_index = rows.video_timestep_index
    if frame_group is not None:
        frame_group = torch.as_tensor(frame_group, dtype=torch.int64).cpu()
        if frame_group.shape != (frames,) or group_delta is None:
            raise ValueError(f"frame_group must be ({frames},) with group_delta")
        used, inverse = torch.unique(frame_group, return_inverse=True)
        base = temb[rows.video_timestep_index].float()
        extra = base + group_delta[used.to(group_delta.device)].to(device=temb.device, dtype=torch.float32)
        frame_index = temb.shape[0] + inverse
        temb = torch.cat([temb, extra.to(temb.dtype)], dim=0)
        _indices, block_segments = frame_modulation_rewrite(rows.block_adaln_indices[0],
                                                            layout.target_video_segment.row_slice, frame_rows, frame_index)
        row_index = frame_index.repeat_interleave(frame_rows).to(device)

    hidden = run_blocks(dit, hidden, temb, block_segments, rotation)

    final = dit.final_layer
    shift, scale = final.adaln_proj(temb)
    states = final.norm(hidden[:, layout.target_video_segment.row_slice])
    states = states * (1.0 + scale[row_index]) + shift[row_index]
    out = final.video_out(states.to(torch.float32))
    return packing._unpack_video_rows(out, layout.target_video).float()


def stream_forward(dit, head: StreamHead, z: torch.Tensor, estimate: torch.Tensor, r: torch.Tensor,
                   text: torch.Tensor, *, first: torch.Tensor | None = None,
                   memory: tuple[torch.Tensor, torch.Tensor] | None = None,
                   audio_generator: torch.Generator | None = None) -> torch.Tensor:
    """One generator pass over the window: ``(1, 24, T, H, W)`` clean estimates ``z + out`` for every frame.

    ``r`` ``(T,)`` per frame (``R`` = done, the caller keeps done frames' old estimate). ``first`` is
    the ``(1, 24, 1, H, W)`` FL2VA condition (normally ``estimate[:, :, :1]``); ``memory`` is
    ``memory_rows(...)``. Flow time is pinned at the noise end, as in ``dir_h3.generator_step``.
    """
    frames = z.shape[2]
    if estimate.shape != z.shape:
        raise ValueError(f"z and estimate differ: {tuple(z.shape)}, {tuple(estimate.shape)}")
    r = torch.as_tensor(r, dtype=torch.int64).cpu()
    if r.shape != (frames,) or int(r.min()) < 0 or int(r.max()) > head.iterations:
        raise ValueError(f"r must be ({frames},) in 0..{head.iterations}, got {r.tolist()}")
    device = dit.video_patch_proj.weight.device
    previous = patchify(estimate.to(device).float()).flatten(1, 3)
    out = window_forward(dit, z, 0.0, text, first=first, memory=memory, mem_proj=head.mem_proj,
                         frame_group=r, group_delta=head.r_embed, video_extra=head.x0_proj(previous),
                         audio_generator=audio_generator)
    return z.float() + out


# ------------------------------------------------------------------------------ state

@dataclass
class StreamState:
    """One window: per-frame noise, estimate and refinement index, plus the memory."""

    z: torch.Tensor                      # (1, 24, T, H, W)
    estimate: torch.Tensor               # (1, 24, T, H, W)
    r: torch.Tensor                      # (T,) int64, R = done
    time: float = 0.0                    # absolute rotary time of frame 0
    has_first: bool = False              # frame 0 is a clean condition (FL2VA first)
    memory: list = field(default_factory=list)

    @property
    def frames(self) -> int:
        return self.z.shape[2]


def start_state(shape, iterations: int, *, first: torch.Tensor | None = None, generator=None,
                device="cuda") -> StreamState:
    """A fresh window of ``shape = (T, H, W)`` latent frames. With ``first`` (a clean one-frame latent,
    e.g. an encoded image) frame 0 is that condition and done; otherwise every frame starts at r = 0."""
    frames, height, width = shape
    check_window(frames)
    z = torch.randn(1, 24, frames, height, width, device=device, generator=generator)
    estimate = torch.zeros_like(z)
    r = torch.zeros(frames, dtype=torch.int64)
    if first is not None:
        estimate[:, :, :1] = first.to(estimate)
        r[0] = iterations
    return StreamState(z, estimate, r, has_first=first is not None)


def advance(dit, head: StreamHead, state: StreamState, text: torch.Tensor, *, memory_time: str = "real",
            audio_seed: int = 0, grad: bool = False) -> torch.Tensor:
    """One forward: every active frame (``r < R``) moves one iteration. Returns the new estimate
    (also stored in ``state`` unless ``grad``; then the caller owns the update)."""
    iterations = head.iterations
    active = state.r < iterations
    if not bool(active.any()):
        return state.estimate
    first = state.estimate[:, :, :1] if state.has_first else None
    memory = memory_rows(state.memory, state.time, memory_time)
    audio = torch.Generator(state.z.device).manual_seed(audio_seed)
    with torch.set_grad_enabled(grad):
        new = stream_forward(dit, head, state.z, state.estimate, state.r, text, first=first, memory=memory,
                             audio_generator=audio)
        mask = active.to(new.device).view(1, 1, -1, 1, 1)
        new = torch.where(mask, new, state.estimate.float())
    if not grad:
        state.estimate = new.detach()
        state.r = torch.where(active, state.r + 1, state.r)
    return new


def ready(state: StreamState, iterations: int) -> bool:
    """Frames 0..5 are done: the leading clip can be emitted and frame 5 can become the condition."""
    return bool((state.r[: SLIDE + 1] >= iterations).all())


def slide(state: StreamState, iterations: int, *, pool: int = 4, capacity: int = 16, generator=None) -> torch.Tensor:
    """Emit frames 0..4 (one VAE clip), keep frame 0 as a coarse memory keyframe, shift by 5, and
    append 5 fresh frames (new noise, estimate 0, r = 0). Returns the emitted ``(1, 24, 5, H, W)``."""
    if not ready(state, iterations):
        raise RuntimeError(f"slide before frames 0..{SLIDE} are done: r = {state.r.tolist()}")
    _b, _c, _t, height, width = state.z.shape
    emitted = state.estimate[:, :, :SLIDE].clone()
    state.memory.append(MemoryEntry(coarse_tokens(emitted[:, :, :1], pool).cpu(),
                                    pooled_centres(height, width, pool), state.time))
    del state.memory[:-capacity]
    fresh = torch.randn(1, 24, SLIDE, height, width, device=state.z.device, generator=generator)
    state.z = torch.cat([state.z[:, :, SLIDE:], fresh], dim=2)
    state.estimate = torch.cat([state.estimate[:, :, SLIDE:], torch.zeros_like(fresh)], dim=2)
    state.r = torch.cat([state.r[SLIDE:], torch.zeros(SLIDE, dtype=torch.int64)])
    state.time += clip_time()
    state.has_first = True
    return emitted


@torch.no_grad()
def stream_sample(dit, head: StreamHead, text: torch.Tensor, *, shape, clips: int, first: torch.Tensor | None = None,
                  seed: int = 0, pool: int = 4, capacity: int = 16, memory_time: str = "real",
                  device="cuda", on_clip=None, tail: bool = False) -> torch.Tensor:
    """Generate ``clips`` VAE clips (17 pixel frames each) with one sliding window.

    Each window runs ``R`` forwards (the fresh tail needs all of them); frames 0 and 1 are carried
    done. Returns the emitted latents ``(1, 24, 5 * clips, H, W)``, plus the two done frames left in
    the window with ``tail`` (``5 * clips + 2`` frames, a length H3's VAE decodes without padding);
    ``on_clip(index, latent)`` sees each clip as it is emitted (decode it there for a live stream).
    """
    generator = torch.Generator(device).manual_seed(seed)
    state = start_state(shape, head.iterations, first=first, generator=generator, device=device)
    emitted = []
    for index in range(clips):
        while not ready(state, head.iterations):
            advance(dit, head, state, text, memory_time=memory_time, audio_seed=seed + 1)
        clip = slide(state, head.iterations, pool=pool, capacity=capacity, generator=generator)
        emitted.append(clip)
        if on_clip is not None:
            on_clip(index, clip)
    if tail:
        if not bool((state.r[:2] >= head.iterations).all()):
            raise RuntimeError(f"tail frames are not done: r = {state.r.tolist()}")
        emitted.append(state.estimate[:, :, :2])
    return torch.cat(emitted, dim=2)


def rows_per_window(shape, *, memory: int, pool: int, text_length: int) -> int:
    """Sequence length of one forward (budgeting): text + memory + first + audio + video."""
    from h3_musubi import audio_frames

    frames, height, width = shape
    per_key = math.ceil(height / 2 / pool) * math.ceil(width / 2 / pool)
    frame_rows = (height // 2) * (width // 2)
    return text_length + memory * per_key + frame_rows + 2 * audio_frames(frames) + frames * frame_rows
