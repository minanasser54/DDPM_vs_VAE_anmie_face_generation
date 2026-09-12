"""
Usage:
    python train_vae.py
    python train_vae.py --epochs 40 --batch_size 128 --kl_weight 1.0
    python train_vae.py --resume checkpoints/vae.pt
"""

import argparse
import os
import time

import torch
from torch.optim import Adam

import conf
import utils
from vae import VAE, vae_loss


def parse_args():
    p = argparse.ArgumentParser(description="Train the VAE on anime faces")
    p.add_argument("--epochs", type=int, default=conf.VAE_EPOCHS)
    p.add_argument("--batch_size", type=int, default=conf.VAE_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=conf.VAE_LR)
    p.add_argument("--kl_weight", type=float, default=conf.VAE_KL_WEIGHT)
    p.add_argument("--latent_dim", type=int, default=conf.VAE_LATENT_DIM)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--data_dir", type=str, default=conf.TRAIN_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    utils.set_seed()
    device = conf.DEVICE
    print(f"Device: {device}")

    dataset, loader = utils.get_dataloader(args.data_dir, args.batch_size, shuffle=True, train=True)
    print(f"Train images: {len(dataset)} | batches/epoch: {len(loader)}")

    model = VAE(latent_dim=args.latent_dim).to(device)
    print(f"VAE params: {utils.count_parameters(model):,}")

    optimizer = Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    best_loss = float("inf")
    if args.resume and os.path.exists(args.resume):
        ckpt = utils.load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("best_loss", best_loss)
        print(f"Resumed from epoch {start_epoch}")

    fixed_batch, _ = next(iter(loader))
    fixed_batch = fixed_batch[:16].to(device)

    os.makedirs(conf.LOG_DIR, exist_ok=True)
    log_path = os.path.join(conf.LOG_DIR, "vae_train.log")
    log_file = open(log_path, "a")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        loss_meter = utils.AverageMeter()
        recon_meter = utils.AverageMeter()
        kl_meter = utils.AverageMeter()

        t0 = time.time()
        for step, (images, _) in enumerate(loader):
            images = images.to(device, non_blocking=True)

            recon, mu, logvar = model(images)
            loss, recon_loss, kl_loss = vae_loss(recon, images, mu, logvar, kl_weight=args.kl_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = images.size(0)
            loss_meter.update(loss.item(), bs)
            recon_meter.update(recon_loss.item(), bs)
            kl_meter.update(kl_loss.item(), bs)

            if step % conf.VAE_LOG_EVERY == 0:
                print(f"epoch {epoch} step {step}/{len(loader)} "
                      f"loss={loss.item():.4f} recon={recon_loss.item():.4f} kl={kl_loss.item():.4f}")

        elapsed = time.time() - t0
        msg = (f"epoch {epoch} DONE | avg_loss={loss_meter.avg:.4f} "
               f"avg_recon={recon_meter.avg:.4f} avg_kl={kl_meter.avg:.4f} "
               f"time={elapsed:.1f}s")
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

        if epoch % conf.VAE_SAMPLE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                samples = model.sample(16, device=device)
                recon, _, _ = model(fixed_batch)
            utils.save_image_grid(samples, os.path.join(conf.SAMPLE_DIR, "vae", f"epoch_{epoch:03d}.png"))
            utils.save_image_grid(
                torch.cat([fixed_batch, recon], dim=0),
                os.path.join(conf.SAMPLE_DIR, "vae", f"recon_epoch_{epoch:03d}.png"),
                nrow=8,
            )
            model.train()

        ckpt = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
            "latent_dim": args.latent_dim,
        }
        utils.save_checkpoint(conf.VAE_CKPT, **ckpt)
        if loss_meter.avg < best_loss:
            best_loss = loss_meter.avg
            ckpt["best_loss"] = best_loss
            utils.save_checkpoint(conf.VAE_CKPT.replace(".pt", "_best.pt"), **ckpt)

    log_file.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
