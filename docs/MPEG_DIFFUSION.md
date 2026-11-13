# MPEG-Diffusion on MiniMax H3

MPEG-Diffusion ([Dennis Menn, Tianpei Gu, Guande He, Aryan V S, Yanke Song, Nan Liu, Desai Xie, Xun Huang, Chenfeng Xu and Diana Marculescu, arXiv 2611.12069](https://arxiv.org/abs/2611.12069)) generates the visually important region at high resolution (HR) and the rest at low resolution (LR). Their sampler (Sec. 4) has four stages:

1. The first `k` of `T` steps denoise the whole video at LR.
2. At step `k`, a linear head `s = wᵀ LN(h)` on a middle block picks the HR region `M`.
3. The clean LR estimate inside `M` is upscaled and re-noised. The remaining steps denoise `M` at HR and the rest at LR, with **dual-resolution attention**: each query reads keys/values (K/V) only at its own resolution, and the missing K/V come from linear maps `W = A + U Vᵀ` (copy/average plus a rank-r delta), applied before RoPE. Their Proposition 1 is the reason: with one shared K/V, a token standing in for `n` tokens is exact only when those tokens are identical.
4. A mixed-resolution decoder (latent upscaler `U` plus a 3D-conv mixer `G`) merges the two latents before the VAE.

They fully fine-tune the DiT together with the K/V maps and a learned resolution embedding (Wan2.1, 1,250 steps on 64 B200s). Each training video is encoded at both resolutions, noised to the same level, and each latent is supervised only on its own region (their eq. 2). `k` is 25 or 40 of 50 steps.

Two implementations are here:

| Script | LR track | Status |
|---|---|---|
| [`lot/two_track_h3.py`](../lot/two_track_h3.py) | The unmodified H3 model at half resolution, as in the paper. HR tokens exist only on `M`. No LoT adapter. | Clean, coherent images with no fine-tuning. Faces are only as good as the LR track draws them (see below). |
| [`lot/mixed_res_h3.py`](../lot/mixed_res_h3.py) | LoT merged tokens from a `train_h3` run, with the three pieces adapted to LoT's single mixed-extent sequence. | Output is smeared: it inherits the LoT adapter's coarse regions, which are already soft without these additions. |

Both use [`lot/mixed_res.py`](../lot/mixed_res.py) (attention, region head, mixer; model-agnostic and covered by CPU tests).

## What `two_track_h3.py` matches and what it does not

| Paper | Here |
|---|---|
| LR = base model at half resolution | Same: 672×384 for a 1344×768 still. H3's rotary positions are normalised by `sqrt(area)`, so both grids share one coordinate frame. |
| Dual-resolution attention, maps before RoPE | Same for HR queries (`two_track_attention`). LR queries see only LR keys, so the LR track is exactly the base model and no `f_down` is needed. |
| Maps ridge-initialised on paired K/V | Same (`fit-kv`, full rank). The fit is weak: relative residual 0.52 on keys and 0.89 on values on 200 images (copying gives 0.61 and 1.07), the same as on 24 images. The paper reports cosine 0.94 for keys and 0.75 for values on Wan. |
| DiT + maps + resolution embedding fine-tuned | A rank-16 LoRA with the same loss (`train_tt.py lora`). A first run on an RTX 5090 was stopped at step 500, with held-out loss down 21% (LR) and 26% (HR); no quality result yet. See [TRAINING.md](TRAINING.md). |
| Region head on object labels (LLM + Grounded SAM) or an LPIPS label | Caption-guided main-subject masks from YOLO's 80 classes plus YuNet faces, trained on the frozen DiT like the paper. See [REGION_HEAD.md](REGION_HEAD.md). The first, latent-L1-labelled head picked rugs and furniture over people. |
| Learned upscaler `U` and mixer `G` | `U` is the VAE (decode, bicubic ×2, encode); `G` is a feathered paste. |
| `k` = 50–80% of steps | `k` = 18 of 20, then 6 extra steps linear in sigma. Earlier switches drift from the LR track without the fine-tune, which shows as seams. |

On the 5090 at 1344×768, 20 steps: full res 1008 tokens/step, half res 252, two-track 281–322 (12–28% of the frame at HR). Wall time is about the same for all three (17–21 s) at `--swap 8`, and two-track runs 24 forwards, so this is a quality testbed, not yet a speedup.


---
[← Back to the README](../README.md) · Write-up: [scrya.com/mpeg_diffusion](https://scrya.com/mpeg_diffusion)
