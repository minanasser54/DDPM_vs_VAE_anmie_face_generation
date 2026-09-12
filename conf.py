import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "train")   # ImageFolder layout: data/train/all/*.jpg
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "test")      # ImageFolder layout: data/test/all/*.jpg

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
SAMPLE_DIR = os.path.join(PROJECT_ROOT, "samples")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

VAE_CKPT = os.path.join(CHECKPOINT_DIR, "vae.pt")
DDPM_CKPT = os.path.join(CHECKPOINT_DIR, "ddpm.pt")


# Dataset
NUM_IMAGES = 100000       
TRAIN_SPLIT = 0.97         

IMAGE_SIZE = 64            
CHANNELS = 3

# Device / reproducibility
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2

# VAE hyperparameters
VAE_LATENT_DIM = 256
VAE_BASE_CHANNELS = 64
VAE_BATCH_SIZE = 256
VAE_EPOCHS = 20
VAE_LR = 2e-4
VAE_KL_WEIGHT = 0.1          
VAE_LOG_EVERY = 100
VAE_SAMPLE_EVERY = 1

# DDPM hyperparameters
DDPM_TIMESTEPS = 1000
DDPM_BETA_START = 1e-4
DDPM_BETA_END = 0.02
DDPM_BASE_CHANNELS = 64       
DDPM_TIME_DIM = 256
DDPM_BATCH_SIZE = 32
DDPM_EPOCHS = 50
DDPM_LR = 2e-4
DDPM_EMA_DECAY = 0.9995
DDPM_EMA_WARMUP_STEPS = 800     # start averaging only after this many steps (avoids EMA anchored to noise-init weights)
DDPM_GRAD_CLIP = 0.7            
DDPM_LOG_EVERY = 100
DDPM_SAMPLE_EVERY = 1
DDPM_SAMPLE_GRID = 24
DDPM_GRAD_ACCUM_STEPS = 2

# Evaluation / benchmarking
EVAL_NUM_SAMPLES = 100
FID_BATCH_SIZE = 64
