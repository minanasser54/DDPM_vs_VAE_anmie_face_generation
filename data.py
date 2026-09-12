"""
Usage:
    python data.py
    python data.py --image_size 64
"""

import argparse
import glob
import os

from PIL import Image
from tqdm import tqdm

import conf


def parse_args():
    p = argparse.ArgumentParser(description="Download + preprocess anime face dataset")
    p.add_argument("--num_images", type=int, default=conf.NUM_IMAGES,
                    help="cap on total images used (train+test). Dataset has ~63.6k total.")
    p.add_argument("--image_size", type=int, default=conf.IMAGE_SIZE)
    p.add_argument("--train_split", type=float, default=conf.TRAIN_SPLIT)
    p.add_argument("--seed", type=int, default=conf.SEED)
    return p.parse_args()


def download_animefaces() -> str:
    import kagglehub
    print("Downloading anime face dataset via kagglehub (cached if already present)...")
    path = kagglehub.dataset_download("splcher/animefacedataset")
    print("Path to dataset files:", path)
    return path


def find_image_dir(root: str) -> str:
    candidates = glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True)
    candidates += glob.glob(os.path.join(root, "**", "*.png"), recursive=True)
    if not candidates:
        raise FileNotFoundError(f"No .jpg/.png files found under {root}. Check the kagglehub download path.")
    return os.path.dirname(candidates[0])


def build_dataset(num_images: int, image_size: int, train_split: float, seed: int):
    import random
    random.seed(seed)

    raw_root = download_animefaces()
    img_dir = find_image_dir(raw_root)
    all_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
    print(f"Found {len(all_files)} total images in {img_dir}")

    num_images = min(num_images, len(all_files))
    random.shuffle(all_files)
    subset = all_files[:num_images]

    n_train = int(num_images * train_split)
    train_files = subset[:n_train]
    test_files = subset[n_train:]

    train_out = os.path.join(conf.TRAIN_DIR, "all")
    test_out = os.path.join(conf.TEST_DIR, "all")
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(test_out, exist_ok=True)

    print(f"Resizing + saving {len(train_files)} train / {len(test_files)} test images "
          f"at {image_size}x{image_size} ...")

    _resize_and_save(train_files, train_out, image_size, "train")
    _resize_and_save(test_files, test_out, image_size, "test")

    print("Done.")
    print(f"  train dir: {conf.TRAIN_DIR}  ({len(train_files)} images)")
    print(f"  test dir:  {conf.TEST_DIR}  ({len(test_files)} images)")


def _resize_and_save(files, out_dir, image_size, desc):
    for i, fpath in enumerate(tqdm(files, desc=f"resize[{desc}]")):
        try:
            img = Image.open(fpath).convert("RGB")
            img = img.resize((image_size, image_size), Image.BICUBIC)
            out_path = os.path.join(out_dir, f"{desc}_{i:06d}.jpg")
            img.save(out_path, quality=95)
        except Exception as e:
            print(f"skipping {fpath}: {e}")


def main():
    args = parse_args()

    if os.path.isdir(conf.TRAIN_DIR) and len(glob.glob(os.path.join(conf.TRAIN_DIR, "all", "*.jpg"))) > 0:
        print(f"{conf.TRAIN_DIR} already populated - skipping rebuild.")
        print("Delete data/train and data/test manually if you want to regenerate.")
        return

    build_dataset(
        num_images=args.num_images,
        image_size=args.image_size,
        train_split=args.train_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
