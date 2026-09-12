"""
Usage:
    python train_ddpm.py
    python train_ddpm.py --batch_size 8 --grad_accum_steps 8   # effective batch size 64, low memory
    python train_ddpm.py --epochs 50 --batch_size 64
    python train_ddpm.py --resume checkpoints/ddpm.pt

Saves:
    checkpoints/ddpm.pt              (latest, resumable, includes EMA weights)
    samples/ddpm/epoch_XXX.png       (samples from the EMA model via full reverse loop)
    logs/ddpm_train.log
"""

import argparse
import copy
import os
import time

import torch
from torch.optim import AdamW

import conf
import utils
from ddpm import UNet, Diffusion


def parse_args():
    p = argparse.ArgumentParser(description="Train the DDPM UNet on anime faces")
    p.add_argument("--epochs", type=int, default=conf.DDPM_EPOCHS)
    p.add_argument("--batch_size", type=int, default=conf.DDPM_BATCH_SIZE,
                    help="physical (per-step) batch size - how many images actually sit in GPU memory at once")
    p.add_argument("--grad_accum_steps", type=int, default=conf.DDPM_GRAD_ACCUM_STEPS,
                    help="number of micro-batches to accumulate gradients over before each optimizer step. "
                         "effective_batch_size = batch_size * grad_accum_steps. Use this to raise the effective "
                         "batch size on a memory-constrained GPU without raising --batch_size.")
    p.add_argument("--lr", type=float, default=conf.DDPM_LR)
    p.add_argument("--timesteps", type=int, default=conf.DDPM_TIMESTEPS)
    p.add_argument("--grad_clip", type=float, default=conf.DDPM_GRAD_CLIP)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--data_dir", type=str, default=conf.TRAIN_DIR)
    p.add_argument("--sample_grid", type=int, default=conf.DDPM_SAMPLE_GRID)
    return p.parse_args()


class EMA:
    """Exponential moving average of model weights. Standard DDPM trick that
    meaningfully improves sample quality/stability - the EMA model is what
    gets used for sampling/eval, while the raw model keeps training normally.
    Averaging only kicks in after DDPM_EMA_WARMUP_STEPS so the EMA isn't
    anchored to the (meaningless) random initialization for too long."""

    def __init__(self, model: torch.nn.Module, decay: float = conf.DDPM_EMA_DECAY,
                 warmup_steps: int = conf.DDPM_EMA_WARMUP_STEPS):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.ema_model = copy.deepcopy(model)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            self.ema_model.load_state_dict(model.state_dict())
            return
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.mul_(self.decay).add_(p, alpha=1 - self.decay)
        for ema_b, b in zip(self.ema_model.buffers(), model.buffers()):
            ema_b.copy_(b)


def main():
    args = parse_args()
    utils.set_seed()
    device = conf.DEVICE
    print(f"Device: {device}")

    effective_batch_size = args.batch_size * args.grad_accum_steps
    print(f"Physical batch_size={args.batch_size} x grad_accum_steps={args.grad_accum_steps} "
          f"-> effective batch_size={effective_batch_size}")

    dataset, loader = utils.get_dataloader(args.data_dir, args.batch_size, shuffle=True, train=True)
    print(f"Train images: {len(dataset)} | micro-batches/epoch: {len(loader)} "
          f"| optimizer steps/epoch: {len(loader) // args.grad_accum_steps}")

    model = UNet(timesteps=args.timesteps).to(device)
    print(f"UNet params: {utils.count_parameters(model):,}")

    diffusion = Diffusion(timesteps=args.timesteps, device=device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    ema = EMA(model)

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = utils.load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        ema.ema_model.load_state_dict(ckpt["ema_model_state"])
        ema.step_count = ckpt.get("ema_step_count", ema.warmup_steps + 1)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from epoch {start_epoch}")

    os.makedirs(conf.LOG_DIR, exist_ok=True)
    log_path = os.path.join(conf.LOG_DIR, "ddpm_train.log")
    log_file = open(log_path, "a")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        loss_meter = utils.AverageMeter()  # tracks the true (unscaled) per-micro-batch loss, for readable logging

        t0 = time.time()
        optimizer.zero_grad()
        for step, (images, _) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            t = diffusion.sample_timesteps(images.size(0))
            x_t, noise = diffusion.q_sample(images, t)

            predicted_noise = model(x_t, t)
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)

            # Scale down before backward so accumulated grads == the average
            # grad of a true `effective_batch_size` batch, not the sum of
            # grad_accum_steps separate full-size updates.
            (loss / args.grad_accum_steps).backward()

            # loss_meter logs the TRUE (unscaled) loss value, so printed
            # numbers stay directly comparable to a run without accumulation.
            loss_meter.update(loss.item(), images.size(0))

            is_accum_boundary = (step + 1) % args.grad_accum_steps == 0
            is_last_step_in_epoch = (step + 1) == len(loader)
            if is_accum_boundary or is_last_step_in_epoch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)  # EMA tracks optimizer steps, not micro-batches

            if step % conf.DDPM_LOG_EVERY == 0:
                print(f"epoch {epoch} step {step}/{len(loader)} mse_loss={loss.item():.4f}")

        elapsed = time.time() - t0
        msg = f"epoch {epoch} DONE | avg_mse_loss={loss_meter.avg:.4f} time={elapsed:.1f}s"
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

        utils.save_checkpoint(
            conf.DDPM_CKPT,
            model_state=model.state_dict(),
            ema_model_state=ema.ema_model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            epoch=epoch,
            timesteps=args.timesteps,
            ema_step_count=ema.step_count,
        )

        if epoch % conf.DDPM_SAMPLE_EVERY == 0 or epoch == args.epochs - 1:
            samples = diffusion.sample(ema.ema_model, n=args.sample_grid, verbose=False)
            utils.save_image_grid(samples, os.path.join(conf.SAMPLE_DIR, "ddpm", f"epoch_{epoch:03d}.png"), nrow=4)

    log_file.close()
    print("Training complete.")


if __name__ == "__main__":
    main()