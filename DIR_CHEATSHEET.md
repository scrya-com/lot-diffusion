# DiR on MiniMax H3: getting started

> **Research code. One 1,000-step run exists. Its speed is real; its pictures are not done.** Progress, renders and losses: [MiniMax-H3 issue #3](https://github.com/johndpope/MiniMax-H3/issues/3). Explainer: [scrya.com/dir_diffusion](https://scrya.com/dir_diffusion).

**DiR (data-space iteration) is ByteDance Seed's method**: Shanchuan Lin, Yansong Peng, Fu-Yun Wang and Haoqi Fan, ["Few-Step Generation via Data-Space Iteration"](https://arxiv.org/abs/2610.12102), arXiv 2610.12102. MiniMax H3 is MiniMax's model. This repo is an independent port to H3 and is not affiliated with either company.

**What it does.** A few-step generator refines its own clean estimate, `x̂0⁽ʳ⁺¹⁾ = G(x̂0⁽ʳ⁾, z, r)`. It uses the same noise `z` every time, with no schedule and no re-noising. It takes R = 4 iterations instead of 20 sampler steps. It is trained with DMD against the frozen H3 and needs **text prompts only, no clean images**. Each step costs one H3 forward:

| 768×1152 still, RTX PRO 4000 Blackwell 24 GB | DiT time |
|---|---|
| dense H3, 20 steps | 35.8 s |
| DiR, 4 iterations | 6.6 s (5.4×) |
| DiR, 1 iteration | 1.65 s |

Run everything from the repo root. Only one GPU job runs at a time: the scripts refuse to start next to another one and never kill it.

---

## 0. Setup (once)

```bash
pip install -r requirements.txt
hf download Comfy-Org/MiniMax-H3 diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors --local-dir models
export LOT_H3_CHECKPOINT=$PWD/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
export LOT_H3_VAE=/path/to/minimax_h3_video_vae_fp16.safetensors          # renders and caching
export LOT_H3_TEXT_ENCODER=/path/to/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors   # caching prompts only
```

All paths live in `lot/lot_paths.py`, and each one can be overridden by an environment variable. H3 itself (int8 loader, block swap, LoRA, VAE) comes from the unmodified musubi-tuner subset in `third_party/`.

**Hardware:** one 24 GB NVIDIA card. Training peaks at about 20.5 GB with `--swap 4` (4 blocks streamed from CPU). A Mac (MPS) port is on the `mac` branch; it is untested on real hardware and much slower.

## 1. Check the install

```bash
python3 lot/test_lot.py                 # CPU, no weights: every test prints, last line "ok N"
python3 lot/parity_dir_h3.py            # GPU: the untrained generator equals H3's one-step prediction -> ok=1
python3 lot/parity_stream_h3.py --swap 4   # GPU: streaming forward == stock T2VA / FL2VA, and window timing -> ok=1
```

Do not train if a parity check fails. With zero-initialised new parts, the untrained generator *is* H3, and these tests prove the plumbing.

## 2. Prepare samples

DMD learns from **prompts**. The image is only used for its shape, and as the first frame when you train streaming video. So a training set is a folder of pictures, each with a same-stem caption:

```
my_stills/
  kitchen_001.jpg   kitchen_001.txt   <- "a photorealistic isometric kitchen at dusk, warm light"
  cafe_002.png      cafe_002.txt
  talk_003.mp4      talk_003.txt      <- clips are fine too (24 fps)
```

- **Captions:** write what you want the generator to be good at, in H3's long descriptive style. Variety beats count; run 1 used 801 isometric 3D stills.
- **Stills:** any size. `--size W H` cover-resizes and centre-crops (multiples of 32), and training then takes random crops (`--max-latent-hw`).
- **Clips (optional):** 24 fps. The first `--frames` frames are kept, on the 17n+5 grid (22 → 7 latent frames). To cut one:

```bash
ffmpeg -v error -y -i talk.mp4 -vf "fps=24,scale=768:768:force_original_aspect_ratio=increase,crop=768:768" \
  -frames:v 22 -an -c:v libx264 -crf 16 -pix_fmt yuv420p my_stills/talk_003.mp4
```

## 3. Save latents and encoded prompts (the cache)

```bash
python3 lot/build_cache_h3.py media --src my_stills --out my_cache --size 768 1152   # H3 video VAE -> latents
python3 lot/build_cache_h3.py text  --out my_cache                                   # Qwen3-VL-32B -> prompt states
```

- `media` writes `<stem>_<W>x<H>_minimaxh3.safetensors` (key `latent_<T>x<H>x<W>`) and merges the captions into `captions.json`.
- `text` loads the text encoder **once**, writes `<stem>_minimaxh3_te.safetensors` (`hidden_states`, `attention_mask`) and is resumable. It takes about 9.5 s per caption, so 800 prompts is about 2 h. Run it alone on the GPU.
- Training uses every pair that has both files. A fixed-seed shuffle sets aside `--holdout` items (default 8) for evals and renders.

Already have H3 latents (`z.pt`)? `build_cache_h3.py latents` skips the VAE (see `lot/CHEATSHEET.md`, section 2).

**Sanity check:** `ls my_cache/*_minimaxh3_te.safetensors | wc -l` should equal the number of samples.

## 4. Train the still-image generator

Dry pass first. It runs one generator update and one fake update with gradients, prints the gradient sums and memory, and writes nothing:

```bash
python3 lot/train_dir_h3.py --check --cache my_cache --swap 4 --max-latent-hw 48 48
```

Every line must end `ok=1`. Then train with run 1's recipe, which is the paper's recipe adapted to batch 2:

```bash
LOT_H3_TRAIN=1 setsid nohup python3 lot/train_dir_h3.py --cache my_cache \
  --swap 4 --max-latent-hw 48 48 --dmd-weight paper --betas 0.9 0.95 --grad-clip 0 \
  --fake-updates 4 --tau-shift beta:2.5 --accum 2 --ema 0.999 \
  --steps 1000 --eval-every 50 --save-every 100 --out lot/runs/my_dir > /tmp/my_dir.log 2>&1 &
```

| Flag | Run 1 | Why |
|---|---|---|
| `--dmd-weight paper` | on | w(τ) = 1: `grad = (x0_fake − x0_real)/τ`, the paper's final setting |
| `--tau-shift beta:2.5` | on | τ ~ Beta(2.5, 1), the paper's importance sampling |
| `--fake-updates 4` | 4 | fake-model updates per generator update, as in the paper |
| `--betas 0.9 0.95` | from step 800 | paper uses β1 = 0 at batch 256; batch 2 needs momentum |
| `--accum 2` | 2 | samples per update (H3 runs one sample at a time) |
| `--ema 0.999` | on | averaged generator with warm-up; saved as `*_ema` |
| `--max-latent-hw 48 48` | on | 768×768 crops keep the step at about 20.5 GB |
| `--rank` | 16 | LoRA rank for generator and fake model; a rank-64 fake is the next experiment |

**Cost:** about 65 s per step on the 24 GB card, so 1,000 steps take about 18 h.

**Watch:**

```bash
grep "kind=dir step" /tmp/my_dir.log | tail -3        # dmd, fake_loss, grad_norm, s_per_step
grep '"eval"' lot/runs/my_dir/log.jsonl | tail -2      # held-out gap to the real model, change per iteration
```

**Pause and resume:** stop the process by PID (never `pkill -f`). Trim `log.jsonl` to steps ≤ N, then rerun the same command with `--resume-step N`. That loads the step-N save. The optimizer restarts, and with β2 = 0.95 it re-warms within about 20 steps.

**Saves** are written to `--out` every `--save-every` steps: `lora_gen_N`, `dir_head_N`, `lora_fake_N`, `dir_fake_head_N`, and with `--ema` also `*_N_ema`.

### Warm start from Minimax-H3-Turbo (experimental)

[LightX2V's Turbo LoRAs](https://github.com/ModelTC/Minimax-H3-Turbo) (Apache-2.0) are far ahead of run 1 ([dir-diffusion#1](https://github.com/scrya-com/dir-diffusion/issues/1)). Converting one gives DiR a much better starting point than base H3's one-step blur:

```bash
hf download lightx2v/Minimax-h3-Turbo minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors --local-dir turbo
python3 lot/turbo_lora.py turbo/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors --rank 128 --out lot/runs/turbo4_init
LOT_H3_TRAIN=1 python3 lot/train_dir_h3.py --init lot/runs/turbo4_init --rank 128 --fake-rank 64 --swap 12 --ema 0 \
  --max-latent-hw 48 48 --dmd-weight paper --betas 0.9 0.95 --grad-clip 0 --fake-updates 4 --tau-shift beta:2.5 --accum 2 \
  --steps 1000 --eval-every 50 --save-every 100 --out lot/runs/dir_turbo
```

- **What converts exactly:** out_proj, fc1 and fc2.
- **What is approximated:** Turbo's fused qkv (rank 384) is cut to rank 128 by SVD, which keeps 92–99.9% of its energy per block.
- **What is dropped:** the 2 token-refiner blocks.
- **Memory:** rank 128 on a 24 GB card needs `--swap 12` and `--ema 0`. The `--check` peak is about 20.5 GB before optimizer state. `--fake-rank` keeps the fake model smaller.

## 5. Render and compare

```bash
python3 lot/render_dir_h3.py --run lot/runs/my_dir --tag _1000 --prompts 3 --swap 4 --size 768 1152
python3 lot/render_dir_h3.py --run lot/runs/my_dir --tag _1000 --ema --prompts 3 --swap 4 --size 768 1152
```

The output is `grid.png`. Columns are dense 20-step H3, then DiR after 1, 2, 3 and 4 iterations, all from the same seed. `meta.json` holds the DiT milliseconds and how much each iteration changed the image. The prompts are held out (`--holdout` from the run's config).

## 6. Streaming video (experimental, not yet trained)

One window of 7 latent frames slides by one VAE clip (17 frames) and is never re-noised. Each window's first frame is the previous window's frame 5, fed as H3's own FL2VA condition. See `lot/stream_dir_h3.py`.

```bash
# warm start from a still run: stream 4 clips from a dense first frame -> p*.mp4 + contact sheets
python3 lot/render_stream_h3.py --run lot/runs/my_dir --tag _1000 --clips 4 --prompts 2 --swap 4

# train the stream (windows start from your cache's stills as first frames)
python3 lot/train_stream_dir_h3.py --check --init lot/runs/my_dir --init-tag _1000 --cache my_cache
LOT_H3_TRAIN=1 python3 lot/train_stream_dir_h3.py --init lot/runs/my_dir --init-tag _1000 \
  --cache my_cache --max-latent-hw 32 32 --steps 500 --out lot/runs/my_stream
```

- **Window cost:** a 7×48×48 window forward takes 3.3 s (3.8 s with 16 memory keyframes) at about 20.6 GB.
- **Known limit:** the DMD teachers see the first frame but not the long-range memory, so training doesn't yet reward long-range consistency.

## Gotchas

- **Render the live weights for early checkpoints.** An EMA started without warm-up is mostly untrained weights for hundreds of steps. Warm-up EMA saves (`ema_warmup: true` in `config.json`) are fine.
- **Wait on a PID, not a pattern.** A `while pgrep -f X` loop matches its own command line and never ends. `kill` by PID.
- **Only one GPU job at a time.** Rendering needs the GPU, so pause training first.
- **No LPIPS or GAN loss is used**, on purpose. The paper uses exactly two losses: DMD and the fake model's flow loss.
- **Differences from the paper:** rank-16 LoRAs instead of a full fine-tune, batch 2 instead of 256, zero init (so step 0 equals H3), and `r` drawn at random rather than step mod R. Details: [issue #3](https://github.com/johndpope/MiniMax-H3/issues/3).
