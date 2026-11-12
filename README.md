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
| [`lot/`](lot) | The method (`layout`, `flow`, `adapter`, `layout_sources`), the H3 contract (`h3`, `h3_positions`, `h3_splice`), the H3 runtime glue (`h3_musubi.py`), and scripts to cache data, fit the extent bank, train, render and time. Start at [`lot/CHEATSHEET.md`](lot/CHEATSHEET.md); the full walkthrough is [`lot/README.md`](lot/README.md). |
| [`third_party/musubi_tuner/`](third_party) | The subset of [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner) that runs H3: model, packing, ConvRot int8 loader, block swap, LoRA network, video VAE, text encoder. Copied unmodified (Apache-2.0); see [`NOTICE`](NOTICE). |
| [`comfyui/`](comfyui) | API-format ComfyUI workflows (still, clip, face-box layout). The nodes are in [ComfyUI-MiniMax-H3-Image-Lane](https://github.com/scrya-com/ComfyUI-MiniMax-H3-Image-Lane). |

`h3_musubi.predict(dit, x, t, text, lot=splice)` runs musubi-tuner's H3 forward with three changes on the target video only: the LoT splice embeds its rows, their rotary positions become rectangle centres, and the splice's extent heads replace `video_out` after the shared norm and AdaLN. Text, audio and the 50 blocks are untouched. With an untrained all-1×1 layout the output equals the dense forward exactly (`lot/parity_h3.py`, `rel_rms=0`).

Layouts tile any token grid, including H3's default 1344×768 (24×42 tokens): a 4×4 cell cut by the edge keeps the largest power-of-two pieces that fit (4×2 on the 2-token strip), so nothing is cropped to a multiple of 4. Layouts are per latent frame (a 22-frame clip is 7 latent frames); every token has a temporal extent of 1, as in the LoT paper.

### Mixed resolution, following MPEG-Diffusion

MPEG-Diffusion ([Menn et al., arXiv 2611.12069](https://arxiv.org/abs/2611.12069)) generates the visually important region at high resolution (HR) and the rest at low resolution (LR). Their sampler (Sec. 4) has four stages:

1. The first `k` of `T` steps denoise the whole video at LR.
2. At step `k`, a linear head `s = wᵀ LN(h)` on a middle block picks the HR region `M`.
3. The clean LR estimate inside `M` is upscaled and re-noised. The remaining steps denoise `M` at HR and the rest at LR, with **dual-resolution attention**: each query reads keys/values (K/V) only at its own resolution, and the missing K/V come from linear maps `W = A + U Vᵀ` (copy/average plus a rank-r delta), applied before RoPE. Their Proposition 1 is the reason: with one shared K/V, a token standing in for `n` tokens is exact only when those tokens are identical.
4. A mixed-resolution decoder (latent upscaler `U` plus a 3D-conv mixer `G`) merges the two latents before the VAE.

They fully fine-tune the DiT together with the K/V maps and a learned resolution embedding (Wan2.1, 1,250 steps on 64 B200s). Each training video is encoded at both resolutions, noised to the same level, and each latent is supervised only on its own region (their eq. 2). `k` is 25 or 40 of 50 steps.

Two implementations are here:

| Script | LR track | Status |
|---|---|---|
| [`lot/two_track_h3.py`](lot/two_track_h3.py) | The unmodified H3 model at half resolution, as in the paper. HR tokens exist only on `M`. No LoT adapter. | Clean, coherent images with no fine-tuning. Faces are only as good as the LR track draws them (see below). |
| [`lot/mixed_res_h3.py`](lot/mixed_res_h3.py) | LoT merged tokens from a `train_h3` run, with the three pieces adapted to LoT's single mixed-extent sequence. | Output is smeared: it inherits the LoT adapter's coarse regions, which are already soft without these additions. |

Both use [`lot/mixed_res.py`](lot/mixed_res.py) (attention, region head, mixer; model-agnostic and covered by CPU tests).

#### What `two_track_h3.py` matches and what it does not

| Paper | Here |
|---|---|
| LR = base model at half resolution | Same: 672×384 for a 1344×768 still. H3's rotary positions are normalised by `sqrt(area)`, so both grids share one coordinate frame. |
| Dual-resolution attention, maps before RoPE | Same for HR queries (`two_track_attention`). LR queries see only LR keys, so the LR track is exactly the base model and no `f_down` is needed. |
| Maps ridge-initialised on paired K/V | Same (`fit-kv`, 24 images, full rank). The fit is weak: relative residual 0.52 on keys and 0.90 on values (copying gives 0.63 and 1.11). The paper reports cosine 0.94 for keys and 0.75 for values on Wan. |
| DiT + maps + resolution embedding fine-tuned | **Not done.** This is the main gap. |
| Region head on object labels (LLM + Grounded SAM) or an LPIPS label | Latent-L1 label: the 25% of tokens where the upscaled LR estimate is furthest from the true HR latent. It tends to pick rugs and furniture texture over people, the failure the paper warns about for perceptual labels. |
| Learned upscaler `U` and mixer `G` | `U` is the VAE (decode, bicubic ×2, encode); `G` is a feathered paste. |
| `k` = 50–80% of steps | `k` = 18 of 20, then 6 extra steps linear in sigma. Earlier switches drift from the LR track without the fine-tune, which shows as seams. |

On the 5090 at 1344×768, 20 steps: full res 1008 tokens/step, half res 252, two-track 281–322 (12–28% of the frame at HR). Wall time is about the same for all three (17–21 s) at `--swap 8`, and two-track runs 24 forwards, so this is a quality testbed, not yet a speedup.

#### Training guidance: weak maps, weak region, weak faces

Faces are the clearest failure. The face in the grid is drawn by the LR track at 672×384, where it covers a few tokens, and the HR track only gets the last low-sigma steps, which add texture but cannot redraw eyes or a mouth. Each item below is grounded in the paper; none has been run here yet.

1. **Put the face in `M` (no training).** The paper labels the region with the main subject (Qwen2.5 + Grounded SAM) because it "better aligns with human attention" than the perceptual label. For people, a face detector is the cheap version. `--box` takes one normalised box now; there is no detector in the repo yet. Dilate the box by a token so the eyes are not at the seam.
2. **Fine-tune with the paper's loss so the switch can move earlier.** The HR track must run from a sigma where structure is still open (the paper switches at 50–80% of steps). Recipe:
   - Pairs: a cached HR latent and its LR latent (`VaeUpscaler.down`), both noised with the same sigma.
   - Mask: the face or subject mask (or the 25% of tokens per frame with the highest label, as in the paper).
   - Loss: their eq. 2, velocity MSE of LR tokens outside `M` plus HR tokens inside `M`, each normalised by its region's size, through `two_track_forward`.
   - Train: a LoRA (rank 16 on qkv/out/fc1/fc2, as in `train_h3.py`) instead of the full fine-tune, together with the `UpMaps` and two zero-initialised resolution-embedding vectors (one for HR tokens, one for LR) added to the video rows.
   - Data: portraits and talking heads (the Nikki cache), so the faces get the gradient.
3. **Better map initialisation before the fine-tune.** The value fit (0.90 residual) is the weak half. Fit on hundreds of items instead of 24, and on the low sigmas the HR track actually sees after the switch (the current fit uses 0.5, 0.8, 0.95). Optionally fit one map per sigma band. The paper's linear relation is measured on the base model without fine-tuning, so this is a fair initialisation check: if the value cosine stays well under their 0.75, the fine-tune has to carry it.
4. **Region head on subject labels.** Train the head (`two_track_h3.py region`) with face/subject masks as BCE targets instead of the latent-L1 label. The paper reports mIoU 0.59–0.67 against such masks with a single linear layer.
5. **Learned upscaler and mixer** (their Sec. H): train `U` and `G` on real stills, with the HR latent inside a random mask and the LR latent outside. Use their eq. 11 loss (latent L1 outside `M`, pixel L1, and LPIPS, up-weighted near the boundary). The VAE round trip used now is soft, and it is what the background and the face's surroundings come from.

Order of value for faces: (1) is free and helps at once; (2) is what lets the HR track fix the face; (3) and (4) make (2) converge faster; (5) is about seams and the background.

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

## References

- Nakayama et al., **Level-of-Token Diffusion**, [arXiv 2610.05816](https://arxiv.org/abs/2610.05816). The method this repository implements.
- Menn, Gu, He, V S, Song, Liu, Xie, Huang (Roblox), Xu and Marculescu (UT Austin), **MPEG-Diffusion: Mixed-Resolution Video Generation**, [arXiv 2611.12069](https://arxiv.org/abs/2611.12069). Source of the dual-resolution attention, region head and mixed-resolution sampler in `lot/mixed_res.py` and `lot/two_track_h3.py`.
- [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner), the H3 runtime vendored in `third_party/`.

Cite the papers above, not this repository.
