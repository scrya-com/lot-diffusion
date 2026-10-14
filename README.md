# lot-diffusion

Level-of-Token Diffusion ([Nakayama et al., arXiv 2610.05816](https://arxiv.org/abs/2610.05816)) for the MiniMax H3 video DiT.

> [!WARNING]
> **Research code and research weights.** LoT output is still visibly worse than dense H3: coarse regions are soft, and faces in talking-head shots distort in fixed layouts. The method needs more training. Help wanted: [discussion](https://github.com/scrya-com/lot-diffusion/discussions/1). Results and renders: [johndpope/MiniMax-H3 issue 1](https://github.com/johndpope/MiniMax-H3/issues/1).

LoT gives the transformer fewer, larger tokens where detail is low and recovers the full-resolution velocity afterwards (the paper's eq. 9, patch-wise asymmetric flow). The VAE latent stays full size; only the transformer sequence shrinks. One H3 DiT forward on a 24 GB card (RTX PRO 4000 Blackwell, run-4 weights, all blocks resident):

| Shape | dense | `center` | `bands` | `uniform2` |
|---|---|---|---|---|
| still 768×1152 | 657 ms | 306 ms (**2.15×**) | 395 ms (1.67×) | 265 ms (2.48×) |
| clip 512×768, 22 frames | 2,154 ms | 844 ms (**2.55×**) | 1,124 ms (1.92×) | 665 ms (3.24×) |
| clip 768×1344, 124 frames (48 blocks streamed) | 52.3 s | 18.3 s (**2.85×**) | 20.4 s (2.57×) | not run |

The VAE decode is not shortened; LoT only touches the transformer. Reproduce with `python3 lot/time_lot_h3.py`.

## What is here

| Path | What |
|---|---|
| [`lot/`](lot) | The method (`layout`, `flow`, `adapter`, `layout_sources`), the H3 contract (`h3`, `h3_positions`, `h3_splice`), the H3 runtime glue (`h3_musubi.py`), and scripts to cache data, fit the extent bank, train, render and time. Also an experimental DiR few-step generator (ByteDance Seed's data-space iteration: `dir_h3.py`, `train_dir_h3.py`, streaming `stream_dir_h3.py`); start at [`DIR_CHEATSHEET.md`](DIR_CHEATSHEET.md). Start at [`lot/CHEATSHEET.md`](lot/CHEATSHEET.md); the full walkthrough is [`lot/README.md`](lot/README.md). |
| [`third_party/musubi_tuner/`](third_party) | The subset of [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner) that runs H3: model, packing, ConvRot int8 loader, block swap, LoRA network, video VAE, text encoder. Copied unmodified (Apache-2.0); see [`NOTICE`](NOTICE). |
| [`comfyui/`](comfyui) | API-format ComfyUI workflows (still, clip, face-box layout). The nodes are in [ComfyUI-MiniMax-H3-Image-Lane](https://github.com/scrya-com/ComfyUI-MiniMax-H3-Image-Lane). |

`h3_musubi.predict(dit, x, t, text, lot=splice)` runs musubi-tuner's H3 forward with three changes on the target video only: the LoT splice embeds its rows, their rotary positions become rectangle centres, and the splice's extent heads replace `video_out` after the shared norm and AdaLN. Text, audio and the 50 blocks are untouched. With an untrained all-1×1 layout the output equals the dense forward exactly (`lot/parity_h3.py`, `rel_rms=0`).

## Quick start

```bash
git clone https://github.com/scrya-com/lot-diffusion && cd lot-diffusion
pip install -r requirements.txt
hf download Comfy-Org/MiniMax-H3 diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors --local-dir models
export LOT_H3_CHECKPOINT=$PWD/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
export LOT_H3_VAE=/path/to/minimax_h3_video_vae_fp16.safetensors
hf download johndpope/MiniMax-H3-LoT --local-dir runs/lot_hf

python3 lot/test_lot.py                                                       # CPU, no weights -> ok 32
python3 lot/parity_h3.py                                                      # GPU: LoT 1x1 == dense -> ok=1
python3 lot/time_lot_h3.py --shape still --trained runs/lot_hf/run4_nikki_distill
python3 lot/time_lot_h3.py --shape clip  --trained runs/lot_hf/run4_nikki_distill
```

Train on your own clips: [`lot/README.md` → Training H3 for LoT from your own mp4s](lot/README.md#training-h3-for-lot-from-your-own-mp4s). In short: `build_cache_h3.py media` (VAE), `build_cache_h3.py text` (Qwen3-VL), `procrustes_h3.py` (extent bank), `train_h3.py --distill 1`. One 24 GB card is enough.

All paths are environment variables; [`lot/lot_paths.py`](lot/lot_paths.py) lists them. `MUSUBI_SRC` defaults to the vendored `third_party/`; point it at a full musubi-tuner checkout to use that instead.

## Weights and license

- Code in `lot/` and `comfyui/`: Apache-2.0 ([`LICENSE`](LICENSE)).
- `third_party/musubi_tuner/`: Apache-2.0, from musubi-tuner, which adapts it from Hugging Face Diffusers; see [`NOTICE`](NOTICE).
- MiniMax H3 weights, and anything trained from them (the published LoT adapters and LoRAs at [HF johndpope/MiniMax-H3-LoT](https://huggingface.co/johndpope/MiniMax-H3-LoT)), are under the [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE): **not for use in the EU, UK, South Korea or USA**, and keep its `LICENSE` and `NOTICE` with any copy.

## Citation

Cite the paper: [Level-of-Token Diffusion, arXiv 2610.05816](https://arxiv.org/abs/2610.05816).
