# DDPM vs VAE: Anime Face Generation

From-scratch (no PyTorch Lightning, no `diffusers`) PyTorch implementations of a
convolutional **VAE** and a **DDPM**, trained on the
[anime face dataset](https://www.kaggle.com/datasets/splcher/animefacedataset)
(~63.6k pre-cropped anime face images, resized to 64x64), built to fit a
Colab free-tier T4 or Kaggle's 2xT4 quota.

- **Code:** https://github.com/minanasser54/DDPM_vs_VAE_anmie_face_generation
- **Pretrained checkpoints:** https://huggingface.co/MinaNasser/DDPM_vs_VAE_anime_face/tree/main/checkpoints
- **Report:** see `report.pdf` / `report.typ` for full write-up, architecture diagrams, and qualitative + quantitative comparison.

---

## Project layout

```
conf.py             central config - every hyperparameter and path lives here
utils.py             shared helpers: dataloader, checkpointing, image grids, seeding
data.py               downloads dataset via kagglehub, resizes, splits train/test
vae.py                VAE model (encoder/decoder) + combined recon+KL loss
ddpm.py               UNet (ResNet blocks + self-attention + sinusoidal time embedding) + diffusion process
train_vae.py          trains the VAE
train_ddpm.py         trains the DDPM (supports gradient accumulation)
eval_vae.py           generates samples/reconstructions, dumps images for benchmarking
eval_ddpm.py          generates samples, dumps images for benchmarking
benchmark.py          computes FID + Inception Score for generated vs real images
main.py               runs the full pipeline locally
main.ipynb            same pipeline as a notebook, for Colab/Kaggle
pyproject.toml        uv project file
requirements.txt      plain pip requirements
logs/                 training logs + benchmark_results.json (this run's numbers)
samples/              sample grids saved during/after training
checkpoints/          NOT committed to git (too large) - see "Pretrained checkpoints" below
```

---

## Reproducing our results

Our reported numbers come from these exact settings (already the defaults in
`conf.py` as committed):

| | VAE | DDPM |
|---|---|---|
| `latent_dim` / — | 256 | — |
| `base_channels` | 64 | 64 |
| `batch_size` (physical) | 256 | 32 |
| `grad_accum_steps` | — | 2 (effective batch 64) |
| `epochs` (config'd / actually run) | 20 / 20 | 50 / **10** (see note below) |
| `lr` | 2e-4 | 2e-4 |
| `kl_weight` | 0.1 | — |
| `timesteps` | — | 1000 (linear beta 1e-4 -> 0.02) |

> **Note on DDPM training length:** the results in this repo's `logs/` and
> `samples/` come from a DDPM checkpoint trained for only **10 of the
> configured 50 epochs** (Colab session limits). `avg_mse_loss` was still
> decreasing at epoch 9 (0.0586 -> 0.0268), so the DDPM numbers reported
> here represent a partially-trained model, not a converged one. See
> `report.pdf` for discussion - despite this, DDPM already outperforms the
> VAE on FID, which is itself a notable result.

### 0. One-time setup: Kaggle API token

`data.py` downloads the dataset via `kagglehub`, which needs a Kaggle API
token:
- **Kaggle notebooks**: already configured, skip this.
- **Colab / local**: go to https://www.kaggle.com/settings -> API -> "Create
  New Token" to download `kaggle.json`, then either:
  - place it at `~/.kaggle/kaggle.json` (`chmod 600`), or
  - set env vars `KAGGLE_USERNAME` and `KAGGLE_KEY`.

### 1. Clone + install

```bash
git clone https://github.com/minanasser54/DDPM_vs_VAE_anmie_face_generation.git
cd DDPM_vs_VAE_anmie_face_generation

# option A: uv (recommended - uses pyproject.toml)
uv sync

# option B: plain pip
pip install -r requirements.txt
```

### 2. Reproduce end-to-end (local, with a CUDA GPU)

```bash
python data.py                                          # download + resize + split
python train_vae.py                                     # ~20 epochs, ~14 min on a T4
python eval_vae.py --num_samples 2000                    # dumps samples/vae_eval/*.png
python train_ddpm.py                                     # 50 epochs configured; ~13 min/epoch on a T4
python eval_ddpm.py --num_samples 2000                   # dumps samples/ddpm_eval/*.png
python benchmark.py --both                               # FID + IS for both, written to logs/benchmark_results.json
```

Or run the whole thing with one command:

```bash
python main.py
```

`--skip_data`, `--vae_only`, `--ddpm_only`, `--skip_benchmark` flags are
available - see `python main.py --help`.

### 3. Reproduce end-to-end (Google Colab)

1. Open a new Colab notebook, set **Runtime > Change runtime type > T4 GPU**.
2. Clone the repo and `cd` into it:
   ```python
   !git clone https://github.com/minanasser54/DDPM_vs_VAE_anmie_face_generation.git
   %cd DDPM_vs_VAE_anmie_face_generation
   !pip install -q -r requirements.txt
   ```
3. Upload your `kaggle.json` (see step 0 above), or set
   `KAGGLE_USERNAME`/`KAGGLE_KEY` directly:
   ```python
   import os
   os.environ['KAGGLE_USERNAME'] = 'your_username'
   os.environ['KAGGLE_KEY'] = 'your_key'
   ```
4. Run each stage as a normal script call, exactly as in the local
   instructions above, just prefixed with `!`:
   ```python
   !python data.py
   !python train_vae.py
   !python eval_vae.py --num_samples 2000
   !python train_ddpm.py
   !python eval_ddpm.py --num_samples 2000
   !python benchmark.py --both
   ```
5. Or open `main.ipynb` directly in Colab (`File > Upload notebook`, or open
   from GitHub via `File > Open notebook > GitHub` and pasting the repo URL)
   for the same flow with inline image previews after each stage.

**Colab/Kaggle session limits:** both training scripts checkpoint every
epoch and are resumable:
```bash
python train_vae.py --resume checkpoints/vae.pt
python train_ddpm.py --resume checkpoints/ddpm.pt
```
so you can split training across multiple sessions without losing progress.
This is exactly how the DDPM checkpoint in this repo was produced (10
epochs across however many sessions it took) - if your session disconnects,
just re-run with `--resume` pointing at the last saved checkpoint.

### 4. Skip training - use my pretrained checkpoints

Pretrained weights are hosted on Hugging Face Hub (too large for git):
https://huggingface.co/MinaNasser/DDPM_vs_VAE_anime_face/tree/main/checkpoints

- `vae_best.pt` (142 MB)
- `ddpm.pt` (427 MB, EMA + raw weights, 10 epochs trained)

Download them into `checkpoints/` and skip straight to evaluation:

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
import shutil, os
os.makedirs('checkpoints', exist_ok=True)
for fname in ['ddpm.pt', 'vae_best.pt']:
    path = hf_hub_download(repo_id='MinaNasser/DDPM_vs_VAE_anime_face', filename=f'checkpoints/{fname}')
    shutil.copy(path, f'checkpoints/{fname}')
"

# then:
python eval_vae.py --checkpoint checkpoints/vae_best.pt --num_samples 2000
python eval_ddpm.py --checkpoint checkpoints/ddpm.pt --num_samples 2000
python benchmark.py --both
```

Note: `data.py` still needs to have been run at least once (to populate
`data/test/all/` with real images) before `benchmark.py` can compute FID,
since FID compares generated images against that real held-out split.

---

## Known caveats when reproducing

- **`EVAL_NUM_SAMPLES`**: this repo's committed `conf.py` now defaults to
  `2000`. An earlier commit had this accidentally left at `20` (a leftover
  from local debugging) - if your FID/IS numbers come out wildly different
  from `logs/benchmark_results.json`, or extremely unstable across re-runs,
  check this value; FID needs at least several hundred samples per side to
  be a meaningful estimate, and 20 is far too few.
- **DDPM training length**: as noted above, the shipped `ddpm.pt` reflects
  only 10/50 configured epochs. Samples are already visibly better than the
  VAE's (see `report.pdf`), but training further would likely improve both
  FID and visual fidelity further, and would give a "true" epoch-50 FID
  rather than an epoch-10 one.
- **Gradient accumulation**: `train_ddpm.py` supports
  `--grad_accum_steps N` so a small physical `--batch_size` (e.g. 8, if
  you're on a smaller/shared GPU) still gets a larger, less noisy effective
  batch size. The as-shipped config uses physical batch 32 x accum 2 =
  effective batch 64.
- **Seeds**: `conf.SEED = 42` is used for data shuffling and model init, but
  exact bit-for-bit reproduction across different GPU models/CUDA/cuDNN
  versions isn't guaranteed (standard caveat for any GPU-trained model) -
  expect the same *qualitative* behavior and similar-magnitude losses/FID,
  not identical numbers to the decimal place.

---

## What each model actually does (short version - see `report.pdf` for the full story)
