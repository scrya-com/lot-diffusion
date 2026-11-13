# Large, many-megapixel images: more than two levels (draft)

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


---
[← Back to the README](../README.md) · Write-up: [scrya.com/mpeg_diffusion](https://scrya.com/mpeg_diffusion)
