"""
Usage:
    python eval_ddpm.py --checkpoint checkpoints/ddpm.pt --num_samples 2000
    python eval_ddpm.py --use_raw_weights   # sample with raw (non-EMA) weights instead
"""

import argparse
import os

import torch
from torchvision.utils import save_image

import conf
import utils
from ddpm import UNet, Diffusion


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained DDPM")
    p.add_argument("--checkpoint", type=str, default=conf.DDPM_CKPT)
    p.add_argument("--num_samples", type=int, default=conf.EVAL_NUM_SAMPLES)
    p.add_argument("--output_dir", type=str, default=os.path.join(conf.SAMPLE_DIR, "ddpm_eval"))
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--timesteps", type=int, default=conf.DDPM_TIMESTEPS)
    p.add_argument("--use_raw_weights", action="store_true")
    p.add_argument("--grid_only", action="store_true")
    return p.parse_args()


def load_model(checkpoint_path: str, device: str, timesteps: int, use_ema: bool = True):
    ckpt = utils.load_checkpoint(checkpoint_path, map_location=device)
    model = UNet(timesteps=timesteps).to(device)
    key = "ema_model_state" if (use_ema and "ema_model_state" in ckpt) else "model_state"
    model.load_state_dict(ckpt[key])
    model.eval()
    print(f"Loaded DDPM ({'EMA' if key == 'ema_model_state' else 'raw'} weights) from {checkpoint_path}")
    return model


@torch.no_grad()
def generate_bulk(model, diffusion, n, out_dir, batch_size, device):
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    while saved < n:
        cur_bs = min(batch_size, n - saved)
        samples = diffusion.sample(model, n=cur_bs, verbose=True)
        samples = utils.denormalize(samples)
        for i in range(cur_bs):
            save_image(samples[i], os.path.join(out_dir, f"sample_{saved + i:06d}.png"))
        saved += cur_bs
        print(f"generated {saved}/{n}")
    print(f"Saved {n} generated images to {out_dir}")


def main():
    args = parse_args()
    device = conf.DEVICE
    model = load_model(args.checkpoint, device, args.timesteps, use_ema=not args.use_raw_weights)
    diffusion = Diffusion(timesteps=args.timesteps, device=device)

    preview = diffusion.sample(model, n=16, verbose=True)
    preview_path = os.path.join(conf.SAMPLE_DIR, "ddpm_eval", "preview_grid.png")
    utils.save_image_grid(preview, preview_path, nrow=4)
    print(f"Saved preview grid to {preview_path}")

    if not args.grid_only:
        generate_bulk(model, diffusion, args.num_samples, args.output_dir, args.batch_size, device)


if __name__ == "__main__":
    main()
