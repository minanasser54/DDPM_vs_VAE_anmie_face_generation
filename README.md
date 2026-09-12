# Anime Face Generation: VAE + DDPM from scratch

A from-scratch (no PyTorch Lightning, no diffusers) implementation of a
convolutional **VAE** and a **DDPM** (Denoising Diffusion Probabilistic
Model), trained on the [anime face dataset](https://www.kaggle.com/datasets/splcher/animefacedataset)
(~63.6k pre-cropped 64x64-ish anime face images). Built to run within a
Colab free-tier T4 or Kaggle's 2xT4 / 30hr-per-week quota.

## Why this dataset

Anime faces are simpler and more uniform than real human faces (CelebA):
fewer lighting/pose variations, cleaner crops, lower source resolution.
That means faster convergence and visibly good samples much earlier in
training - good for iterating on a free GPU quota.

## Project layout

```
conf.py            central config - all hyperparameters and paths live here
utils.py            shared helpers: dataloader, checkpointing, image grids, seeding
data.py              downloads dataset via kagglehub, resizes, splits train/test
vae.py               VAE model (encoder/decoder) + combined recon+KL loss
ddpm.py              UNet (ResNet blocks + self-attention + sinusoidal time embedding) + diffusion process
train_vae.py         trains the VAE
train_ddpm.py        trains the DDPM
eval_vae.py          generates samples/reconstructions from a trained VAE, dumps images for benchmarking
eval_ddpm.py         generates samples from a trained DDPM, dumps images for benchmarking
benchmark.py         computes FID + Inception Score for generated vs real images
main.py              runs the full pipeline locally (python main.py)
main.ipynb           same pipeline as a notebook, for Colab/Kaggle
pyproject.toml       uv project file
requirements.txt     plain pip requirements
```

## Quickstart (Colab / Kaggle)

Upload this whole project folder, or clone it, then in a notebook cell:

```bash
!pip install -q -r requirements.txt
# or: !pip install -q uv && !uv pip install -q -r pyproject.toml --system
```

Set up your Kaggle API token (needed for `kagglehub` to download the
dataset) - either upload `kaggle.json` to `~/.kaggle/kaggle.json`, or set
`KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables. On Kaggle notebooks
this is already configured for you.

Then run each stage as a script, exactly as you'd run it from a terminal:

```bash
!python data.py
!python train_vae.py
!python eval_vae.py
!python train_ddpm.py
!python eval_ddpm.py
!python benchmark.py --both
```

See `main.ipynb` for the same flow with live progress + inline image
previews after each stage.

## Quickstart (local)

```bash
uv sync            # or: pip install -r requirements.txt
python main.py      # runs the full pipeline: data -> train_vae -> eval_vae -> train_ddpm -> eval_ddpm -> benchmark
```

Or run stages individually, same commands as above without the `!`.

## The VAE: generation-focused loss

This VAE is built to **generate**, not just reconstruct. That distinction
matters for the loss:

```
total_loss = recon_loss + kl_weight * kl_loss
```

- `recon_loss`: per-pixel MSE, **mean** reduction (not sum) - keeps it at an
  O(0.1-1) scale regardless of resolution.
- `kl_loss`: closed-form `KL(N(mu, sigma^2) || N(0, I))`, also mean-reduced.
- `kl_weight` (`conf.VAE_KL_WEIGHT`, default **1.0**): this is set high on
  purpose. A VAE only produces good samples from `z ~ N(0, I)` if its
  encoder's aggregate posterior actually matches that prior - a low KL
  weight gives you sharp *reconstructions* but a latent space you can't
  actually sample from (random `z` decodes to garbage). Weighting KL this
  heavily trades some reconstruction sharpness for a genuinely sample-able
  latent space, which is the right trade when the deliverable is generation.

**What to expect during training** (`train_vae.py` prints both terms
every log step and epoch average):
- `recon_loss` should steadily fall then plateau.
- `kl_loss` may rise briefly off of near-zero before settling at a stable,
  nonzero value - that's healthy, it means the latent space is doing work.
- `kl_loss` collapsing to ~0 and staying there = posterior collapse (lower
  `VAE_KL_WEIGHT`). `kl_loss` growing unbounded while recon stalls = KL
  dominating too hard (lower `VAE_KL_WEIGHT` a bit).
- Rough numbers at the shipped defaults (batch 128, lr 2e-4, kl_weight 1.0,
  latent_dim 128): total loss ~0.5-0.7 at epoch 0, dropping to ~0.15-0.25 by
  epoch ~10-15; recognizable (if soft) faces in samples from epoch ~3-5.

## The DDPM: architecture + training notes

`ddpm.py` implements a standard epsilon-prediction UNet:
- Pre-activation ResNet blocks (`GroupNorm -> SiLU -> Conv`), with the
  timestep embedding injected into **every** block.
- Self-attention at the 16x16 and 8x8 resolutions.
- A proper sinusoidal timestep embedding table (precomputed for all
  timesteps, then passed through a small MLP), matching the original DDPM
  paper's design.
- A simple **linear** beta schedule (`1e-4 -> 0.02` over 1000 steps) - easy
  to reason about, well-proven, sufficient for a 64x64 dataset like this.
- Trained with the "simple loss" from Ho et al. 2020: plain MSE between
  predicted and true noise, no VLB weighting terms.

Two things in `train_ddpm.py` that matter for actually converging on a
from-scratch setup like this:
- **Gradient clipping** (`conf.DDPM_GRAD_CLIP = 1.0`) - guards against loss
  spikes that can otherwise derail training, especially in the first few
  hundred steps.
- **EMA (exponential moving average) of weights**, with a warmup period
  before averaging kicks in (`conf.DDPM_EMA_WARMUP_STEPS`) so the EMA isn't
  anchored to the meaningless random initialization. Sampling/eval always
  uses the EMA weights - noticeably cleaner samples than the raw model.

**What to expect during training** (defaults: batch 64, lr 2e-4, base
channels 64):
- Per-step loss is inherently noisy (each step samples a random timestep;
  denoising near t=999 is a harder task than near t=0) - judge convergence
  by the **epoch average**, not individual steps.
- `avg_mse_loss` should start around **0.9-1.1** (an untrained model
  predicting noise from noise is close to unit variance) and drop below
  **~0.05-0.08 within the first 3-5 epochs**, then slowly improve toward
  **~0.02-0.04** over the following 20-30 epochs.
- If `avg_mse_loss` is still above ~0.3 after 5 epochs, something's off -
  check `logs/ddpm_train.log`, check the learning rate, check that the data
  actually loaded (not all-black/corrupted images).
- Sample previews (`samples/ddpm/epoch_XXX.png`) should progress: pure
  static -> blurry color blobs (~epoch 2-4) -> vaguely face-shaped
  (~epoch 8-12) -> recognizable anime faces (~epoch 20+).

## Benchmarking

`benchmark.py` computes **FID** (via `pytorch-fid`, the standard
InceptionV3-pool3-features Frechet distance) and **Inception Score** (via
`torchmetrics`) comparing generated images against the held-out real test
split:

```bash
python eval_vae.py --num_samples 2000     # dumps samples/vae_eval/*.png
python eval_ddpm.py --num_samples 2000    # dumps samples/ddpm_eval/*.png
python benchmark.py --both                # scores both against data/test/all
```

Results are appended to `logs/benchmark_results.json` and printed as a
summary table.

## Notes on Colab/Kaggle free-tier fit

- All default settings in `conf.py` are chosen to comfortably fit a single
  T4 (16GB) at 64x64 resolution with room to spare.
- DDPM sampling (ancestral, 1000 steps) is the slow part of the pipeline -
  this is inherent to vanilla DDPM, not a bug. Reduce `--num_samples` in
  `eval_ddpm.py` if you're short on time, or pass `--timesteps` to sample
  with a shorter (coarser) schedule.
- Checkpoints save every epoch and are resumable via `--resume <path>` on
  both training scripts, so you can split training across multiple Colab/
  Kaggle sessions without losing progress.

## Submitting your results

After training, push `checkpoints/`, `samples/`, and `logs/` (including
`benchmark_results.json`) to your GitHub repo alongside this code.
