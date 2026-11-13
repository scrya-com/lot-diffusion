# mpeg-diffusion

Mixed-resolution generation for the MiniMax H3 video DiT: MPEG-Diffusion two-track sampling ([Dennis Menn, Tianpei Gu, Guande He, Aryan V S, Yanke Song, Nan Liu, Desai Xie, Xun Huang, Chenfeng Xu and Diana Marculescu, arXiv 2611.12069](https://arxiv.org/abs/2611.12069)), which spends full-resolution tokens only on a selected region, and Level-of-Token Diffusion ([Kiyohiro Nakayama, Brian Chao, Jan Ackermann, Hansheng Chen, Federico Tombari, Leonidas Guibas, Lior Yariv and Gordon Wetzstein, arXiv 2610.05816](https://arxiv.org/abs/2610.05816)), which merges tokens where detail is low. This repository was called `lot-diffusion`; GitHub redirects the old URLs. Page: [scrya.com/mpeg_diffusion](https://scrya.com/mpeg_diffusion).

> [!WARNING]
> **Research code and research weights.** LoT output is still visibly worse than dense H3: coarse regions are soft, and faces in talking-head shots distort in fixed layouts. The method needs more training. Help wanted: [discussion](https://github.com/scrya-com/mpeg-diffusion/discussions/1). Results and renders: [johndpope/MiniMax-H3 issue 1](https://github.com/johndpope/MiniMax-H3/issues/1).

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

MPEG-Diffusion ([Dennis Menn, Tianpei Gu, Guande He, Aryan V S, Yanke Song, Nan Liu, Desai Xie, Xun Huang, Chenfeng Xu and Diana Marculescu, arXiv 2611.12069](https://arxiv.org/abs/2611.12069)) generates the visually important region at high resolution (HR) and the rest at low resolution (LR). Their sampler (Sec. 4) has four stages:

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

Faces are the clearest failure. The face in the grid is drawn by the LR track at 672×384, where it covers a few tokens, and the HR track only gets the last low-sigma steps, which add texture but cannot redraw eyes or a mouth. Each item below is grounded in the paper. Items 2–5 are implemented (`lot/fetch_pexels.py`, `lot/build_pairs_tt.py`, `lot/train_tt.py`, one command: `lot/train_tt_all.sh`; walkthrough in [`lot/CHEATSHEET.md` section 7](lot/CHEATSHEET.md#7-two-track-mixed-resolution-the-papers-own-arrangement-train-it)) and pass a 30-step smoke test on a 5090. No full training run has been done yet, so there is no quality result to report.

1. **Put the face in `M` (no training).** The paper labels the region with the main subject (Qwen2.5 + Grounded SAM) because it "better aligns with human attention" than the perceptual label. For people, a face detector is the cheap version. `--box` takes one normalised box now; there is no detector in the repo yet. Dilate the box by a token so the eyes are not at the seam.
2. **Fine-tune with the paper's loss so the switch can move earlier.** The HR track must run from a sigma where structure is still open (the paper switches at 50–80% of steps). Recipe:
   - Pairs (`build_pairs_tt.py`): a cached HR latent and its LR latent (decode, downsample ×0.5, encode), noised to the same sigma, plus a face mask (OpenCV YuNet) and a person mask (YOLO segmentation) on the LR token grid.
   - Mask: the face or subject mask (or the 25% of tokens per frame with the highest label, as in the paper).
   - Loss: their eq. 2, velocity MSE of LR tokens outside `M` plus HR tokens inside `M`, each normalised by its region's size, through `two_track_forward`.
   - Train (`train_tt.py lora`): a LoRA (rank 16 on qkv/out/fc1/fc2) instead of the full fine-tune, together with the `UpMaps` and two zero-initialised resolution-embedding vectors (one for HR tokens, one for LR) added to the video rows.
   - Data: people photos with captions. `fetch_pexels.py` downloads them from the Hugging Face list `gaunernst/pexels-portrait` (URLs plus captions, Pexels license). Those are mostly large faces, so mix in scenes with small faces, the failing case.
3. **Better map initialisation before the fine-tune.** The value fit (0.90 residual) is the weak half. `fit-kv --pairs` now fits from the pairs (no VAE round trip per item) over sigmas 0.2–0.95, which covers the sigmas the HR track sees after an earlier switch; use `--items 200` or more. On 4 items the residual was 0.49 on keys and 0.85 on values; on 200 items (Pexels people and iso3d scenes, 5 sigmas) it was 0.52 and 0.89, unchanged, so the limit is the linear map, not the data. The paper's linear relation is also measured on the base model without fine-tuning, so the residual is a fair check: whatever the linear map cannot carry, the LoRA has to.
4. **Region head on subject labels.** `train_tt.py region` trains the head with face and main-subject masks as BCE targets instead of the latent-L1 label, and reports held-out mIoU. Like the paper, it trains on the frozen DiT (`--train` optionally reads a fine-tuned model's states instead). How it works is below. The paper reports 0.59–0.67 against such masks with a single linear layer; the 16-item smoke run gave 0.57.
5. **Learned upscaler and mixer** (their Sec. H): `train_tt.py decoder` trains `U` and `G` on real pairs, with the HR latent inside the mask and the LR latent outside. It uses their eq. 11 loss (latent L1 outside `M` up-weighted at the edge, pixel L1 and LPIPS on a decoded crop at the edge). Gradients pass through the frozen VAE decoder; a 256 px crop peaks at 12 GB. `two_track_h3.py render --decoder` uses `U` at the switch and at the end, and `G` instead of the feathered paste.

Order of value for faces: (1) is free and helps at once; (2) is what lets the HR track fix the face; (3) and (4) make (2) converge faster; (5) is about seams and the background.

#### The region head: how it trains

The region head is what makes the HR region self-selected: at the switch step it reads the DiT's own features and decides, per token, what is the subject. The paper's observation (Sec. 3.3) is that mid-way through denoising, the DiT already encodes where the subject is, so one linear layer can read it out: `s = wᵀ LN(h)` on a middle block, about 5,400 parameters. Training is a linear probe on the frozen model, in two phases (`train_tt.py region`):

**Phase 1: collect features (one forward per image).** For each pair:

1. Noise the half-resolution latent to the sigma of a switch step. With `--switches 6 8 10 12 14`, each pair is seen at one of them in turn, so one head works for an early or a late switch.
2. Run one forward of the frozen H3 at half resolution, with no gradients (252 video tokens plus the text and audio rows at 1344×768).
3. Keep block 25's output for every video token (a 5,376-number vector), with that token's label: 1 on the face or the main subject, else 0.

This is about 0.6 s per image on an RTX 5090 with 8 blocks streamed from CPU. Vectors are stored in fp16 (about 2.7 MB per image).

**Phase 2: fit the head (seconds per epoch).** Logistic regression on the stored vectors: BCE between `s` and the labels, with positives up-weighted by their rarity. No gradient touches H3. Every epoch reports held-out mIoU of the thresholded region (smoothed, `m + α(p − m)`, the paper's rule), split by person/other subject and by switch step. The paper reports 0.59–0.67 against subject masks.

| | LoRA (`train_tt.py lora`) | Region head (`train_tt.py region`) |
|---|---|---|
| Trainable parameters | 82.7 M | about 5.4 k |
| Passes through H3 | forward + backward through 50 blocks, every step | one forward per image, once |
| Sequence | LR + HR tokens | LR tokens only |
| Time on a 5090 | 3–4 s per step | about 0.6 s per image, then seconds |

**Labels for any subject.** `build_pairs_tt.py --subject main` (or `--relabel` on existing pairs) stands in for the paper's LLM + Grounded SAM. It takes YOLO instance masks of any of its 80 classes, prefers the classes the caption names ("a dog on a sofa" picks the dog), and ranks by confidence, size and centrality. Faces come from YuNet. On the 403 landscape iso3d scenes it picked person 274 times, then couch 49, chair 14, potted plant 12, car 9, motorcycle 5, dog 4 and others; 11 had no subject.

**See what it picks.** `two_track_h3.py layout --region head.pt --cache-list CACHE... --switch 10` samples each held-out caption with the base model at half resolution, scores the tokens at the switch, and draws each row as two panels. The left panel is the image with the score heatmap and the selected region outlined. The right panel is the LoT layout that region implies on the full-resolution token grid (1×1 inside, 2×2 on a one-token ring, 4×4 elsewhere), with its token count. The images are generated from the captions, not the training photos.

#### Large, many-megapixel images: more than two levels (draft)

Two levels fit H3's default 1344×768 (1 megapixel). At larger sizes even the half-resolution track gets expensive, and a nested pyramid starts to pay. Level 0 is the whole frame at a size H3 already draws well, and each further level doubles the resolution on a smaller, nested region: half resolution everywhere, full resolution on the subject, ×2 again on the face. Discussion, numbers and help wanted: **[issue #3](https://github.com/scrya-com/mpeg-diffusion/issues/3)**. Draft code: branch [`draft/pyramid-levels`](https://github.com/scrya-com/mpeg-diffusion/tree/draft/pyramid-levels) (`lot/pyramid.py`, `lot/pyramid_h3.py`).

What decides whether a canvas can use more levels:

- **The coarsest level must be a resolution H3 draws well untrained.** At 2688×1536 (4.1 MP), three levels start from 672×384, today's working LR track. At 1344×768 a third level would start from 336×192, far outside H3's training.
- **The token grid must halve evenly.** 1344×768 is 24×42 tokens, and 42/4 is not whole, so only two levels fit. 2688×1536 (48×84) and 3840×2176 (68×120) fit three.
- **The finest queries still read every fine key.** That is what makes them exact (their Proposition 1), and it is why the paper's 14B model is slower than one baseline. Extra levels cut queries and MLP work, not that term.
- **The paper's own large case uses two levels with a wider gap.** At 960p it keeps two tracks with a 240p LR (16 HR tokens per LR token) and reports 4.8× faster denoising than full resolution, at DOVER 0.789 against 0.806.

Counts from `python3 lot/pyramid_h3.py plan --size W H` (CPU; centred nested regions, 300 text/audio rows). These are token and attention-pair counts, not timings:

| Canvas | Megapixels | Levels | Area at each level | Tokens vs full res | Attention pairs vs full res |
|---|---|---|---|---|---|
| 1344×768 | 1.0 | 2 | ½ everywhere, full on 25% | 51% | 38% |
| 2688×1536 | 4.1 | 2 | ½ everywhere, full on 25% | 51% | 34% |
| 2688×1536 | 4.1 | 3 | ¼ everywhere, ½ on 50%, full on 15% | 34% | 20% |
| 3840×2176 | 8.4 | 3 | ¼ everywhere, ½ on 50%, full on 15% | 37% | 21% |

Most of the drop from two to three levels comes from the smaller full-resolution area, not from the extra level itself. The draft passes CPU tests (`L = 2` equals `two_track_attention` exactly; nesting; coarse rows never see finer ones). It has not run on a GPU, and nothing is trained for three levels. Proposed order: train and evaluate two levels (#2), make the speedup show in wall time, then run three levels at 2688×1536.

## Quick start

```bash
git clone https://github.com/scrya-com/mpeg-diffusion && cd mpeg-diffusion
pip install -r requirements.txt
hf download Comfy-Org/MiniMax-H3 diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors --local-dir models
export LOT_H3_CHECKPOINT=$PWD/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
export LOT_H3_VAE=/path/to/minimax_h3_video_vae_fp16.safetensors
hf download johndpope/MiniMax-H3-LoT --local-dir runs/lot_hf

python3 lot/test_lot.py                                                       # CPU, no weights -> ok 33
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

- Kiyohiro Nakayama, Brian Chao, Jan Ackermann, Hansheng Chen, Federico Tombari, Leonidas Guibas, Lior Yariv and Gordon Wetzstein, **Level-of-Token Diffusion**, [arXiv 2610.05816](https://arxiv.org/abs/2610.05816). The method this repository implements.
- Dennis Menn, Tianpei Gu, Guande He, Aryan V S, Yanke Song, Nan Liu, Desai Xie, Xun Huang, Chenfeng Xu and Diana Marculescu (UT Austin, Roblox), **MPEG-Diffusion: Mixed-Resolution Video Generation**, [arXiv 2611.12069](https://arxiv.org/abs/2611.12069). Source of the dual-resolution attention, region head and mixed-resolution sampler in `lot/mixed_res.py` and `lot/two_track_h3.py`.
- [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner), the H3 runtime vendored in `third_party/`.

Cite the papers above, not this repository.
