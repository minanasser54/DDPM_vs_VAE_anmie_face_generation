import os
import random
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.utils as vutils
from torchvision.datasets import ImageFolder

import conf


def set_seed(seed: int = conf.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transforms(image_size: int = conf.IMAGE_SIZE, train: bool = True):
    """
    Images normalized to [-1, 1] (matches tanh VAE decoder output and the
    DDPM's noise-prediction target space). Light horizontal flip augmentation
    on train only - anime faces are usually not symmetric-critical, and this
    doubles effective data variety for free.
    """
    tfs = [T.Resize(image_size), T.CenterCrop(image_size)]
    if train:
        tfs.append(T.RandomHorizontalFlip())
    tfs += [T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    return T.Compose(tfs)


def get_dataloader(split_dir: str, batch_size: int, shuffle: bool = True,
                    image_size: int = conf.IMAGE_SIZE, num_workers: int = conf.NUM_WORKERS,
                    train: bool = True):
    dataset = ImageFolder(root=split_dir, transform=get_transforms(image_size, train=train))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, drop_last=shuffle,
    )
    return dataset, loader


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1] for saving/viewing images."""
    return (x.clamp(-1, 1) + 1) / 2


def save_image_grid(images: torch.Tensor, path: str, nrow: int = 8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    grid = vutils.make_grid(denormalize(images), nrow=nrow)
    vutils.save_image(grid, path)


def save_checkpoint(path: str, **kwargs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(kwargs, path)
    print(f"[checkpoint] saved -> {path}")


def load_checkpoint(path: str, map_location=None):
    map_location = map_location or conf.DEVICE
    ckpt = torch.load(path, map_location=map_location)
    print(f"[checkpoint] loaded <- {path}")
    return ckpt


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
