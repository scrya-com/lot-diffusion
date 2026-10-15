#!/usr/bin/env python3
"""Convert a Minimax-H3-Turbo ComfyUI LoRA into a musubi ``lora_gen.safetensors`` for ``train_dir_h3.py --init``.

    python3 lot/turbo_lora.py /path/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors \
        --rank 128 --out lot/runs/turbo4_init

LightX2V's Turbo LoRAs (github.com/ModelTC/Minimax-H3-Turbo, Apache-2.0) target the same H3 modules as
``h3_musubi.make_lora``: ``blocks.N.attn.qkv_proj`` / ``out_proj`` / ``mlp.fc1`` / ``mlp.fc2`` (fc1 already
in musubi's ``[gate; value]`` order), plus the two token-refiner blocks, which our LoRA does not
target and which are dropped (reported). out/fc1/fc2 are rank 128 and copy exactly. The ComfyUI
qkv is three rank-128 LoRAs fused block-diagonally (rank 384); it is cut to ``--rank`` by a
truncated SVD of its delta, and the energy kept per block is reported (about 0.93-0.999 at 128).

Every output module has ``alpha = rank`` (scale 1) with the Turbo scale folded into the weights,
so ``make_lora(dit, rank)`` loads it unchanged. The output dir holds only ``lora_gen.safetensors``
(no ``dir_head.pt``): the DiR head starts at zero, so at step 0 the generator is Turbo's own first
step at sigma = 1 at every iteration.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

_NAME = re.compile(r"^diffusion_model\.blocks\.(\d+)\.(attn\.qkv_proj|attn\.out_proj|mlp\.fc1|mlp\.fc2)$")


def factor_delta(down: torch.Tensor, up: torch.Tensor, scale: float, rank: int):
    """``(down', up', energy)`` with ``up' @ down'`` the best rank-``rank`` approximation of
    ``scale * up @ down`` (exact, scale folded in, when the input rank is already <= ``rank``)."""
    down, up = down.float(), up.float()
    if down.shape[0] <= rank:
        pad = rank - down.shape[0]
        down = torch.cat([down * scale, down.new_zeros(pad, down.shape[1])])
        up = torch.cat([up, up.new_zeros(up.shape[0], pad)], dim=1)
        return down, up, 1.0
    q_up, r_up = torch.linalg.qr(up)                 # up = Qu Ru, down^T = Qd Rd
    q_down, r_down = torch.linalg.qr(down.T)
    u, s, vh = torch.linalg.svd(scale * (r_up @ r_down.T))
    root = s[:rank].sqrt()
    new_up = (q_up @ u[:, :rank]) * root
    new_down = root[:, None] * (vh[:rank] @ q_down.T)
    energy = float((s[:rank] ** 2).sum() / (s ** 2).sum().clamp(min=1e-30))
    return new_down, new_up, energy


def convert(path: Path, rank: int) -> tuple[dict, dict]:
    from safetensors import safe_open

    out, report = {}, {"dropped": [], "energy": {}}
    with safe_open(str(path), "pt") as handle:
        names = sorted({k.rsplit(".", 2)[0] for k in handle.keys() if k.endswith("lora_A.weight")})
        for name in names:
            match = _NAME.match(name)
            if match is None:
                report["dropped"].append(name)
                continue
            down = handle.get_tensor(f"{name}.lora_A.weight")
            up = handle.get_tensor(f"{name}.lora_B.weight")
            scale = float(handle.get_tensor(f"{name}.alpha")) / down.shape[0]
            new_down, new_up, energy = factor_delta(down, up, scale, rank)
            key = f"lora_unet_blocks_{match.group(1)}_{match.group(2).replace('.', '_')}"
            out[f"{key}.lora_down.weight"] = new_down.to(torch.bfloat16).contiguous()
            out[f"{key}.lora_up.weight"] = new_up.to(torch.bfloat16).contiguous()
            out[f"{key}.alpha"] = torch.tensor(float(rank))
            if energy < 1.0:
                report["energy"][key] = energy
    return out, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("lora", type=Path, help="Turbo ComfyUI LoRA (.safetensors)")
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--out", type=Path, required=True, help="directory for lora_gen.safetensors")
    args = parser.parse_args()
    from safetensors.torch import save_file

    weights, report = convert(args.lora, args.rank)
    modules = len(weights) // 3
    if modules != 200:
        raise SystemExit(f"converted {modules} modules, expected 200 (50 blocks x 4)")
    args.out.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(args.out / "lora_gen.safetensors"),
              {"rank": str(args.rank), "source": args.lora.name, "lot_h3": "turbo_init"})
    energies = sorted(report["energy"].values())
    print(f"LOT_H3 kind=turbo_convert modules={modules} rank={args.rank} dropped={len(report['dropped'])} "
          f"qkv_energy_min={energies[0]:.3f} median={energies[len(energies) // 2]:.3f} out={args.out} ok=1", flush=True)


if __name__ == "__main__":
    main()
