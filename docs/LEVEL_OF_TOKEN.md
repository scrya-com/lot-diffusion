# Level-of-Token (LoT) on MiniMax H3

Level-of-Token Diffusion is by Kiyohiro Nakayama, Brian Chao, Jan Ackermann, Hansheng Chen, Federico Tombari, Leonidas Guibas, Lior Yariv and Gordon Wetzstein ([arXiv 2610.05816](https://arxiv.org/abs/2610.05816)).

> [!WARNING]
> LoT output is still visibly worse than dense H3: coarse regions are soft, and faces in talking-head shots distort in fixed layouts. It needs more training. Results and renders: [johndpope/MiniMax-H3 issue 1](https://github.com/johndpope/MiniMax-H3/issues/1).

LoT gives the transformer fewer, larger tokens where detail is low and recovers the full-resolution velocity afterwards (the paper's eq. 9, patch-wise asymmetric flow). The VAE latent stays full size; only the transformer sequence shrinks. One H3 DiT forward on a 24 GB card (RTX PRO 4000 Blackwell, run-4 weights, all blocks resident):

| Shape | dense | `center` | `bands` | `uniform2` |
|---|---|---|---|---|
| still 768×1152 | 657 ms | 306 ms (**2.15×**) | 395 ms (1.67×) | 265 ms (2.48×) |
| clip 512×768, 22 frames | 2,154 ms | 844 ms (**2.55×**) | 1,124 ms (1.92×) | 665 ms (3.24×) |
| clip 768×1344, 124 frames (48 blocks streamed) | 52.3 s | 18.3 s (**2.85×**) | 20.4 s (2.57×) | not run |

The VAE decode is not shortened; LoT only touches the transformer. Reproduce with `python3 lot/time_lot_h3.py`.

`h3_musubi.predict(dit, x, t, text, lot=splice)` runs musubi-tuner's H3 forward with three changes on the target video only: the LoT splice embeds its rows, their rotary positions become rectangle centres, and the splice's extent heads replace `video_out` after the shared norm and AdaLN. Text, audio and the 50 blocks are untouched. With an untrained all-1×1 layout the output equals the dense forward exactly (`lot/parity_h3.py`, `rel_rms=0`).

Layouts tile any token grid, including H3's default 1344×768 (24×42 tokens): a 4×4 cell cut by the edge keeps the largest power-of-two pieces that fit (4×2 on the 2-token strip), so nothing is cropped to a multiple of 4. Layouts are per latent frame (a 22-frame clip is 7 latent frames); every token has a temporal extent of 1, as in the LoT paper.

## Train it

From your own clips: [`lot/README.md` → Training H3 for LoT from your own mp4s](../lot/README.md#training-h3-for-lot-from-your-own-mp4s). In short: `build_cache_h3.py media` (VAE), `build_cache_h3.py text` (Qwen3-VL), `procrustes_h3.py` (extent bank), `train_h3.py --distill 1`. One 24 GB card is enough. Weights: [johndpope/MiniMax-H3-LoT](https://huggingface.co/johndpope/MiniMax-H3-LoT).


---
[← Back to the README](../README.md) · Write-up: [scrya.com/mpeg_diffusion](https://scrya.com/mpeg_diffusion)
