#!/usr/bin/env python3
"""Publish a two-track LoRA run (``train_tt.py lora``) to the Hugging Face Hub, checkpoint by checkpoint.

    python3 lot/upload_tt_hf.py --lora-dir RUN/lora --repo johndpope/MiniMax-H3-MPEG-Diffusion --folder run1
    python3 lot/upload_tt_hf.py ... --watch     # wait for the final checkpoint, then upload it
    python3 lot/upload_tt_hf.py ... --all       # also the intermediate checkpoints (lora_500, ...)

By default only the final checkpoint is published. Each upload puts, under ``--folder`` in ``--repo``:

- ``lora{tag}.safetensors``: the rank-16 LoRA (kohya keys, as ``train_tt.py`` saves it).
- ``tt_state{tag}.safetensors``: the fine-tuned ``f_up`` maps (``maps.*``) and the two resolution
  vectors (``res_embed.lr``, ``res_embed.hr``), with ``layers`` / ``rank`` / ``hidden`` metadata.
  Converted from ``tt_state{tag}.pt`` so nothing pickled is published. ``two_track_h3.load_trained``
  reads either.
- ``train_log.jsonl`` and ``README.md`` (model card with the held-out eval table so far).

The repo keeps its own ``LICENSE`` and ``NOTICE`` at the root (MiniMax H3 Community License).
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import torch

CODE = "https://github.com/scrya-com/mpeg-diffusion"
PAPER = "https://arxiv.org/abs/2611.12069"
AUTHORS = ("Dennis Menn, Tianpei Gu, Guande He, Aryan V S, Yanke Song, Nan Liu, Desai Xie, Xun Huang, "
           "Chenfeng Xu and Diana Marculescu (UT Austin and Roblox)")


def convert_state(src: Path, dst: Path) -> None:
    from safetensors.torch import save_file

    saved = torch.load(src, map_location="cpu", weights_only=False)
    flat = {f"maps.{k}": v.contiguous() for k, v in saved["maps"].items()}
    flat.update({f"res_embed.{k}": v.contiguous() for k, v in saved["res_embed"].items()})
    save_file(flat, str(dst), {"layers": str(saved["layers"]), "rank": str(saved["rank"]),
                               "hidden": str(saved["hidden"]), "format": "two_track_h3.tt_state"})


def convert_region(src: Path, dst: Path) -> dict:
    """``region_head_tt.pt`` -> safetensors with ``hidden`` / ``block`` / ``switch`` metadata; returns its info."""
    from safetensors.torch import save_file

    saved = torch.load(src, map_location="cpu", weights_only=False)
    info = {k: saved.get(k) for k in ("hidden", "block", "switch", "steps", "label", "held_miou")}
    save_file({k: v.contiguous() for k, v in saved["state"].items()}, str(dst),
              {k: str(v) for k, v in info.items() if v is not None})
    return info


def evals(log: Path) -> list[dict]:
    rows = []
    for line in log.read_text().splitlines():
        if '"eval_lr"' in line:
            rows.append(json.loads(line))
    return rows


def card(repo: str, folder: str, tags: list[str], log: Path, steps_total: int, data: str, region: dict | None = None) -> str:
    rows = evals(log)
    table = "\n".join(f"| {r['step']} | {r['eval_lr']:.4f} | {r['eval_hr']:.4f} |" for r in rows) or "| — | — | — |"
    last = max((json.loads(l)["step"] for l in log.read_text().splitlines() if '"loss"' in l), default=0)
    files = "\n".join(f"- `lora{t}.safetensors` + `tt_state{t}.safetensors` (step {t[1:] if t else 'final'})"
                      for t in tags)
    status = "finished" if "" in tags else f"in progress (step {last} of {steps_total})"
    if region is not None:
        files += (f"\n- `region_head_tt.safetensors`: the self-selected region head (their Sec. 4.3), a linear layer on "
                  f"block {region['block']}'s LR states at switch step {region['switch']} of {region['steps']}, "
                  f"trained on {region['label']} masks with this LoRA on. Held-out mIoU {float(region['held_miou']):.3f} "
                  f"(the paper reports 0.59-0.67 against subject masks).")
    region_flag = " --region weights/{folder}/region_head_tt.safetensors".format(folder=folder) if region else \
        " --box 0.3 0.15 0.7 0.9"
    base = f"https://huggingface.co/{repo}/blob/main"
    return f"""# MiniMax H3 two-track (MPEG-Diffusion) LoRA, run 1

> [!WARNING]
> **Research checkpoint. Training {status}.** No quality evaluation yet beyond the held-out loss below.
> Help wanted: [{CODE}/issues/2]({CODE}/issues/2).

> [!CAUTION]
> Derived from MiniMax H3 under the [MiniMax H3 Community License]({base}/LICENSE) (see [`NOTICE`]({base}/NOTICE)):
> use and redistribution **only outside the European Union, the United Kingdom, the Republic of Korea and the United
> States of America**. Do not use these weights to impersonate anyone; disclose machine-generated output.

Weights for two-track mixed-resolution sampling on the MiniMax H3 DiT, following **MPEG-Diffusion: Mixed-Resolution
Video Generation** by {AUTHORS}, [arXiv 2611.12069]({PAPER}). This is an independent port, not their release.

The unmodified H3 renders the whole frame at half resolution, a region (for example the face) gets full-resolution
tokens, and dual-resolution attention lets each query read keys and values at its own resolution. These files are
what the paper fine-tunes so the full-resolution track can start earlier: a rank-16 LoRA on every block's
qkv / out / fc1 / fc2, the `f_up` K/V maps (ridge-initialised, then trained) and two resolution-embedding vectors.

## Files

{files}
- `train_log.jsonl`: every step, plus held-out evals every 250 steps.

## Use

```bash
git clone {CODE} && cd mpeg-diffusion
hf download {repo} --include "{folder}/*" --local-dir weights
python3 lot/two_track_h3.py render --out out --train weights/{folder} --switch 12 --hr-steps 0 \\
   {region_flag} --outline
```

Base: `minimax_h3_fl2va_pruned_int8_convrot.safetensors` from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3).

## Training

{data}

Loss: their eq. 2 in H3's velocity form: LR velocity MSE outside the region plus HR velocity MSE inside it, each
averaged over its own area. Region: face and person masks (union), else a random box (20% of steps use a box).
Sigma: shift 3. AdamW 8-bit: LoRA 1e-4, maps 1e-4, resolution vectors 1e-3. Batch 1, {steps_total} steps,
one RTX 5090, 8 blocks streamed from CPU. Code: `lot/train_tt.py lora` in [{CODE}]({CODE}).

## Held-out loss (8 pairs, sigmas 0.3 / 0.6 / 0.9)

| step | LR outside region | HR inside region |
|---|---|---|
{table}

Lower is better. A falling loss shows the model fits the objective; it is not a quality result. Renders and
comparisons will be posted on [issue 2]({CODE}/issues/2).
"""


def upload(args, tags: list[str]) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for tag in tags:
            shutil.copy(args.lora_dir / f"lora{tag}.safetensors", stage / f"lora{tag}.safetensors")
            convert_state(args.lora_dir / f"tt_state{tag}.pt", stage / f"tt_state{tag}.safetensors")
        shutil.copy(args.lora_dir / "log.jsonl", stage / "train_log.jsonl")
        region = None
        if args.region is not None and args.region.is_file():
            region = convert_region(args.region, stage / "region_head_tt.safetensors")
        (stage / "README.md").write_text(card(args.repo, args.folder, sorted(set(args.done + tags), key=_order),
                                              args.lora_dir / "log.jsonl", args.steps, args.data, region))
        if args.dry_run is not None:
            shutil.copytree(stage, args.dry_run, dirs_exist_ok=True)
        else:
            api.upload_folder(repo_id=args.repo, folder_path=str(stage), path_in_repo=args.folder,
                              commit_message=f"{args.folder}: {', '.join(t or 'final' for t in tags) or 'card'}")
    args.done = sorted(set(args.done + tags), key=_order)
    print(f"LOT_H3 kind=hf_upload repo={args.repo} folder={args.folder} tags={tags} "
          f"dry_run={args.dry_run is not None} ok=1", flush=True)


def _order(tag: str) -> int:
    return int(tag[1:]) if tag else 10**9


def ready(args) -> list[str]:
    """Checkpoints on disk that are complete (both files) and not uploaded yet."""
    found = []
    for path in args.lora_dir.glob("tt_state*.pt"):
        tag = path.stem[len("tt_state"):]
        if not args.all and tag:
            continue
        if (args.lora_dir / f"lora{tag}.safetensors").is_file() and tag not in args.done:
            if time.time() - path.stat().st_mtime > 30:          # let the writer finish
                found.append(tag)
    return sorted(found, key=_order)


def region_card(repo: str, folder: str, info: dict, log: Path | None, data: str) -> str:
    """Model card for a region head published on its own."""
    import re

    best_line, per = "", ""
    if log is not None and log.is_file():
        text = log.read_text()
        match = re.search(r"tt_region_best epoch=(\d+)", text)
        if match:
            line = re.search(rf"tt_region_epoch epoch={match.group(1)} .*", text)
            best_line = line.group(0) if line else ""
        else:                                                   # logs before best-epoch selection: last epoch
            lines = re.findall(r"tt_region_epoch epoch=\d+ .*", text)
            best_line = lines[-1] if lines else ""
        pairs = re.findall(r"(class:\w+|switch:\d+)=([0-9.]+)\(n=(\d+)\)", best_line)
        per = "\n".join(f"| {k.replace('class:', 'subject: ').replace('switch:', 'switch step ')} | {v} | {n} |"
                         for k, v, n in pairs)
    base = f"https://huggingface.co/{repo}/blob/main"
    switches = info.get("switches") or [info.get("switch")]
    return f"""# Region head: where the subject is, from MiniMax H3's own features

> [!WARNING]
> **Research checkpoint.** It finds the subject in most test prompts but not all. See *Limits* below.
> Help and results: [scrya-com/mpeg-diffusion issue #2](https://github.com/scrya-com/mpeg-diffusion/issues/2).

> [!CAUTION]
> Derived from MiniMax H3 under the [MiniMax H3 Community License]({base}/LICENSE) (see [`NOTICE`]({base}/NOTICE)):
> use only outside the European Union, the United Kingdom, the Republic of Korea and the United States of America.

The self-selected region of MPEG-Diffusion (Sec. 4.3) by Dennis Menn, Tianpei Gu, Guande He, Aryan V S, Yanke Song,
Nan Liu, Desai Xie, Xun Huang, Chenfeng Xu and Diana Marculescu (UT Austin and Roblox),
[arXiv 2611.12069](https://arxiv.org/abs/2611.12069), ported to MiniMax H3. Independent port, not their release.

`s = wᵀ LN(h)`: one linear layer (about 5,400 weights) on block {info.get('block')}'s half-resolution video tokens,
partway through sampling. It scores each token; smoothing and the paper's threshold `m + α(p − m)` give the region,
and the region gives a Level-of-Token layout (full-size tokens inside, 2×2 on a ring, 4×4 elsewhere). The DiT is
frozen; nothing else is trained.

## Files

- `region_head_tt.safetensors`: the head, with `hidden` / `block` / `switch` metadata.

## Use

```bash
git clone https://github.com/scrya-com/mpeg-diffusion && cd mpeg-diffusion
hf download {repo} --include "{folder}/*" --local-dir weights
# Draw what it picks, and the LoT layout that implies, for held-out captions:
python3 lot/two_track_h3.py layout --region weights/{folder}/region_head_tt.safetensors --out out --switch 10
# Or let it pick the full-resolution region in two-track sampling:
python3 lot/two_track_h3.py render --region weights/{folder}/region_head_tt.safetensors --out out --outline
```

## Training

{data}

Phase 1: one forward of the frozen H3 per picture at half resolution, at one of the switch steps {switches} of
{info.get('steps')} (cycled), keeping block {info.get('block')}'s vector per token. 70% of pictures get a random
off-centre crop and a horizontal flip, so position does not predict the label. Phase 2: logistic regression on the
stored vectors; the epoch with the best held-out mIoU is kept. Code: `lot/train_tt.py region`
([how it works](https://github.com/scrya-com/mpeg-diffusion/blob/main/docs/REGION_HEAD.md)).

## Held-out mIoU (64 pictures)

Overall: **{float(info.get('held_miou') or 0):.3f}** (the paper reports 0.59–0.67 against subject masks on Wan).

| Split | mIoU | Pictures |
|---|---|---|
{per}

## Limits

- The training data is mostly people (Pexels portraits and isometric scenes), so it is best at finding people.
- The held-out set has very few non-person subjects, so the "other" score is not reliable.
- Labels come from detectors (YuNet faces, YOLOE prompted with the caption's words), not from humans.
"""


def upload_region(args) -> None:
    from huggingface_hub import HfApi

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        info = convert_region(args.region, stage / "region_head_tt.safetensors")
        saved = torch.load(args.region, map_location="cpu", weights_only=False)
        info["switches"] = saved.get("switches")
        (stage / "README.md").write_text(region_card(args.repo, args.folder, info, args.log, args.data))
        if args.dry_run is not None:
            shutil.copytree(stage, args.dry_run, dirs_exist_ok=True)
        else:
            HfApi().upload_folder(repo_id=args.repo, folder_path=str(stage), path_in_repo=args.folder,
                                  commit_message=f"{args.folder}: region head (held-out mIoU {float(info['held_miou']):.3f})")
    print(f"LOT_H3 kind=hf_region_upload repo={args.repo} folder={args.folder} dry_run={args.dry_run is not None} ok=1",
          flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lora-dir", type=Path, default=None)
    parser.add_argument("--region-only", action="store_true",
                        help="publish --region on its own into --folder (no LoRA), with its own card")
    parser.add_argument("--log", type=Path, default=None, help="--region-only: the training log, for per-split mIoU")
    parser.add_argument("--repo", default="johndpope/MiniMax-H3-MPEG-Diffusion")
    parser.add_argument("--folder", default="run1")
    parser.add_argument("--steps", type=int, default=2000, help="planned steps (for the card)")
    parser.add_argument("--data", default="Data: 1,951 Pexels photos of people (HF list `gaunernst/pexels-portrait`, "
                        "Pexels license) and 403 isometric 3D stills, all at 1344x768, as HR/LR latent pairs with "
                        "YuNet face and YOLO person masks.")
    parser.add_argument("--watch", action="store_true", help="wait for the final checkpoint, upload, stop")
    parser.add_argument("--all", action="store_true", help="also upload intermediate checkpoints")
    parser.add_argument("--dry-run", type=Path, default=None, help="stage into this dir, upload nothing")
    parser.add_argument("--region", type=Path, default=None, help="region_head_tt.pt from train_tt.py region, published beside the LoRA")
    parser.add_argument("--poll", type=int, default=120)
    args = parser.parse_args()
    if args.region_only:
        if args.region is None:
            raise SystemExit("--region-only needs --region")
        upload_region(args)
        return
    if args.lora_dir is None:
        raise SystemExit("--lora-dir is required (or --region-only)")
    args.done = []
    while True:
        tags = ready(args)
        if tags:
            upload(args, tags)
        if not args.watch or "" in args.done:
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
