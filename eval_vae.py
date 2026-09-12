"""
Usage:
    python eval_vae.py --checkpoint checkpoints/vae.pt --num_samples 2000
    python eval_vae.py --interpolate
"""

import argparse
import os

import torch
from tqdm import tqdm

import conf
import utils
from vae import VAE


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained VAE")
    p.add_argument("--checkpoint", type=str, default=conf.VAE_CKPT)
    p.add_argument("--num_samples", type=int, default=conf.EVAL_NUM_SAMPLES)
    p.add_argument("--output_dir", type=str, default=os.path.join(conf.SAMPLE_DIR, "vae_eval"))
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--interpolate", action="store_true")
    p.add_argument("--grid_only", action="store_true", help="skip bulk generation, just save preview grids")
    return p.parse_args()


def load_model(checkpoint_path: str, device: str):
    ckpt = utils.load_checkpoint(checkpoint_path, map_location=device)
    latent_dim = ckpt.get("latent_dim", conf.VAE_LATENT_DIM)
    model = VAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def generate_bulk(model, n, out_dir, batch_size, device):
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    pbar = tqdm(total=n, desc="generating VAE samples")
    while saved < n:
        cur_bs = min(batch_size, n - saved)
        samples = model.sample(cur_bs, device=device)
        samples = utils.denormalize(samples)
        for i in range(cur_bs):
            from torchvision.utils import save_image
            save_image(samples[i], os.path.join(out_dir, f"sample_{saved + i:06d}.png"))
        saved += cur_bs
        pbar.update(cur_bs)
    pbar.close()
    print(f"Saved {n} generated images to {out_dir}")


@torch.no_grad()
def interpolate(model, dataset, device, steps=10):
    x1 = dataset[0][0].unsqueeze(0).to(device)
    x2 = dataset[1][0].unsqueeze(0).to(device)

    mu1, _ = model.encoder(x1)
    mu2, _ = model.encoder(x2)

    alphas = torch.linspace(0, 1, steps, device=device)
    zs = torch.stack([mu1.squeeze(0) * (1 - a) + mu2.squeeze(0) * a for a in alphas])
    imgs = model.decoder(zs)

    out_path = os.path.join(conf.SAMPLE_DIR, "vae_eval", "interpolation.png")
    utils.save_image_grid(imgs, out_path, nrow=steps)
    print(f"Saved interpolation grid to {out_path}")


def main():
    args = parse_args()
    device = conf.DEVICE
    model = load_model(args.checkpoint, device)
    print(f"Loaded VAE from {args.checkpoint}")

    with torch.no_grad():
        preview = model.sample(64, device=device)
    preview_path = os.path.join(conf.SAMPLE_DIR, "vae_eval", "preview_grid.png")
    utils.save_image_grid(preview, preview_path, nrow=8)
    print(f"Saved preview grid to {preview_path}")

    if args.interpolate:
        _, test_loader = utils.get_dataloader(conf.TEST_DIR, batch_size=1, shuffle=False, train=False)
        interpolate(model, test_loader.dataset, device)

    if not args.grid_only:
        generate_bulk(model, args.num_samples, args.output_dir, args.batch_size, device)


if __name__ == "__main__":
    main()
