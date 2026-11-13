# The region head: how it trains

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


---
[← Back to the README](../README.md) · Write-up: [scrya.com/mpeg_diffusion](https://scrya.com/mpeg_diffusion)
