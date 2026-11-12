"""DRAFT: more than two resolution levels, nested (a generalisation of MPEG-Diffusion's two tracks).

MPEG-Diffusion (Dennis Menn, Tianpei Gu, Guande He, Aryan V S, Yanke Song, Nan Liu, Desai Xie, Xun
Huang, Chenfeng Xu and Diana Marculescu; UT Austin and Roblox; arXiv:2611.12069) runs two tracks:
a complete low-resolution (LR) grid and high-resolution (HR) tokens on a region, and each query
reads keys/values at its own resolution (their Proposition 1 is why). This module extends that to
``L`` nested levels, each twice the resolution of the one before:

- Level 0 is complete (every token of the coarsest grid exists).
- Level ``l >= 1`` exists only where ``masks[l - 1]`` refines a level ``l - 1`` token into its
  2x2 children. Masks are nested: a token can be refined only where it exists.
- A level-``l`` query reads a *complete* level-``l`` grid of keys/values: its own level's tokens
  where they exist, and elsewhere the level ``l - 1`` grid mapped up by ``f_up`` (copy plus a
  rank-``r`` delta, ``mixed_res.UpMaps``), recursively from level 0. Text/audio and level-0 queries
  see only text, audio and level 0, so level 0 is the unmodified model at its resolution.

With ``L = 2`` this is ``mixed_res.two_track_attention`` exactly (a CPU test checks it).

What more levels do and do not buy (see ``token_share`` and ``attention_cost``):

- Fewer tokens: with 25% of the frame at full resolution, two levels use 0.25 + 0.25 = 50% of the
  full-resolution tokens; three levels (quarter everywhere, half on 50%, full on 15%) use
  0.0625 + 0.125 + 0.15 = 34%.
- Not cheaper attention for the finest queries: they still read a complete full-resolution grid of
  keys (that is what makes them exact). The paper's 14B model is slower than a baseline for this
  reason. Extra levels shrink queries and MLP work, not that term.
- The coarsest level must be a resolution the base model already draws well, and the token grid
  must halve evenly ``L - 1`` times. At 1344x768 the full token grid is 24x42, so only two levels
  fit (42 / 4 is not whole); 2688x1536 (48x84) fits three, with a 672x384 coarsest level.

Status: draft. CPU-tested attention and plan; ``pyramid_h3.py`` wires it into H3 but has not run
on a GPU, and nothing is trained for more than two levels.
"""

from __future__ import annotations

import torch

from mixed_res import UpMaps, _rope, _sdpa

EXTENT = (1, 2, 2)


def cell_order(time: int, height: int, width: int, scale: int = 2) -> torch.Tensor:
    """Permutation from "each coarse token, then its ``scale^2`` cells row-major" to the raster order
    of the ``(time, height * scale, width * scale)`` grid (``mixed_res.TwoTrackPlan.cell_order``)."""
    t, u, v = torch.meshgrid(torch.arange(time), torch.arange(height), torch.arange(width), indexing="ij")
    du, dv = torch.meshgrid(torch.arange(scale), torch.arange(scale), indexing="ij")
    cells = (t.reshape(-1, 1) * (height * scale) + u.reshape(-1, 1) * scale + du.reshape(1, -1)) * (width * scale) \
        + v.reshape(-1, 1) * scale + dv.reshape(1, -1)
    return torch.argsort(cells.flatten())


class PyramidPlan:
    """Index tensors for ``L`` nested levels.

    ``base`` is the level-0 token grid ``(T, h, w)``. ``masks[l - 1]`` is a bool ``(T, h * 2^(l-1),
    w * 2^(l-1))`` on the level ``l - 1`` grid: the tokens refined into level ``l``. The video rows of
    the packed sequence are level 0 (raster), then each level's existing tokens (raster).
    """

    def __init__(self, base: tuple[int, int, int], masks: list[torch.Tensor], scale: int = 2):
        time, height, width = (int(n) for n in base)
        self.scale = int(scale)
        self.shapes = [(time, height, width)]
        present = [torch.ones(time, height, width, dtype=torch.bool)]
        for level, mask in enumerate(masks, start=1):
            mask = mask.bool().cpu()
            if tuple(mask.shape) != self.shapes[-1]:
                raise ValueError(f"masks[{level - 1}] has shape {tuple(mask.shape)}, level {level - 1} grid is "
                                 f"{self.shapes[-1]}")
            if (mask & ~present[-1]).any():
                raise ValueError(f"masks[{level - 1}] refines tokens that level {level - 1} does not have")
            t, h, w = self.shapes[-1]
            self.shapes.append((t, h * self.scale, w * self.scale))
            present.append(mask.repeat_interleave(self.scale, 1).repeat_interleave(self.scale, 2))
        self.present = present
        self.index = [torch.nonzero(p.flatten()).squeeze(1) for p in present]
        self.orders = [None] + [cell_order(*shape, self.scale) for shape in self.shapes[:-1]]

    @property
    def levels(self) -> int:
        return len(self.shapes)

    @property
    def counts(self) -> list[int]:
        return [int(i.numel()) for i in self.index]

    @property
    def grid_counts(self) -> list[int]:
        return [t * h * w for t, h, w in self.shapes]

    def offsets(self, video_start: int = 0) -> list[int]:
        """First row of each level in the packed sequence."""
        out, row = [], video_start
        for count in self.counts:
            out.append(row)
            row += count
        return out

    def to(self, device) -> "PyramidPlan":
        self.index = [i.to(device) for i in self.index]
        self.orders = [None] + [o.to(device) for o in self.orders[1:]]
        return self


def pyramid_attention(attn, hidden: torch.Tensor, rotation: torch.Tensor, *, layer: int, plan: PyramidPlan,
                      maps: list[UpMaps | None] | None, video_start: int, grid_rotations: list[torch.Tensor],
                      capture: dict | None = None) -> torch.Tensor:
    """Dual-resolution attention over ``L`` nested levels (module docstring).

    ``hidden`` is ``(1, S, D)``: text/audio rows, then the video rows in ``plan`` order. ``rotation``
    is the table of those rows. ``grid_rotations[l - 1]`` is the table of the *complete* level-``l``
    grid (raster), for ``l >= 1``. ``maps[l - 1]`` maps level ``l - 1`` keys/values up to level ``l``
    (``None``: copy). ``capture`` receives level 0's pre-RoPE keys/values (for fitting the maps).
    """
    batch, length, _ = hidden.shape
    query, key, value = attn.qkv_proj(hidden).split(attn.inner_dim, dim=-1)
    query = attn.q_norm(query.reshape(batch, length, attn.num_heads, attn.head_dim))
    key = attn.k_norm(key.reshape(batch, length, attn.num_heads, attn.head_dim))
    value = value.reshape(batch, length, attn.num_heads, attn.head_dim).contiguous()
    offsets, counts = plan.offsets(video_start), plan.counts
    split = video_start + counts[0]
    if capture is not None:
        capture[layer] = (key[:, video_start:split].detach(), value[:, video_start:split].detach())
    query_r = _rope(query, rotation).to(value)
    key_r = _rope(key, rotation).to(value)
    output = hidden.new_zeros(batch, length, attn.inner_dim, dtype=value.dtype)
    output[:, :split] = _sdpa(query_r[:, :split], key_r[:, :split], value[:, :split])

    grid_key, grid_value = key[:, video_start:split], value[:, video_start:split]   # complete level-0 grid
    cells = plan.scale * plan.scale
    for level in range(1, plan.levels):
        if counts[level] == 0:
            break                                          # nested: no deeper level exists either
        up = None if maps is None else maps[level - 1]
        if up is None:
            grid_key = grid_key.repeat_interleave(cells, dim=1)
            grid_value = grid_value.repeat_interleave(cells, dim=1)
        else:
            grid_key = up.expand(layer, EXTENT, "k", grid_key).to(key.dtype)
            grid_value = up.expand(layer, EXTENT, "v", grid_value).to(value.dtype)
        start, stop = offsets[level], offsets[level] + counts[level]
        order, index = plan.orders[level], plan.index[level]
        grid_key = grid_key[:, order].index_copy(1, index, key[:, start:stop])
        grid_value = grid_value[:, order].index_copy(1, index, value[:, start:stop])
        rotated = _rope(grid_key, grid_rotations[level - 1]).to(value)
        output[:, start:stop] = _sdpa(query_r[:, start:stop],
                                      torch.cat([key_r[:, :video_start], rotated], dim=1),
                                      torch.cat([value[:, :video_start], grid_value], dim=1))
    return attn.out_proj(output)


# ------------------------------------------------------------------------ planning numbers


def max_levels(token_h: int, token_w: int, scale: int = 2, min_side: int = 1) -> int:
    """How many nested levels a full-resolution token grid allows: the level-0 grid must be whole."""
    levels = 1
    while token_h % scale == 0 and token_w % scale == 0 and min(token_h, token_w) // scale >= min_side:
        token_h, token_w, levels = token_h // scale, token_w // scale, levels + 1
    return levels


def token_share(area_shares: list[float], scale: int = 2) -> float:
    """Tokens relative to the full-resolution grid. ``area_shares[l]`` is the share of the frame
    covered at level ``l`` (level 0 is 1.0, the last level is full resolution)."""
    last = len(area_shares) - 1
    return sum(share / (scale * scale) ** (last - level) for level, share in enumerate(area_shares))


def attention_cost(plan: PyramidPlan, prefix: int = 0) -> dict[str, float]:
    """Query-key pairs per attention layer, against one full-resolution pass of the same canvas.

    Level 0 and the prefix attend over prefix + level 0; a level-``l`` query attends over prefix +
    the complete level-``l`` grid."""
    full = prefix + plan.grid_counts[-1]
    pairs = (prefix + plan.counts[0]) ** 2
    pairs += sum(count * (prefix + grid) for count, grid in zip(plan.counts[1:], plan.grid_counts[1:]))
    tokens = sum(plan.counts)
    return {"tokens": tokens, "token_share": tokens / plan.grid_counts[-1], "pairs": pairs,
            "pair_share": pairs / full ** 2, "finest_queries_keys": prefix + plan.grid_counts[-1]}


__all__ = ["PyramidPlan", "pyramid_attention", "cell_order", "max_levels", "token_share", "attention_cost"]
