"""
benchmark.py - computes FID (Frechet Inception Distance) and Inception Score
for a folder of generated images against the real anime-face test split.

Uses `pytorch-fid` for FID (the canonical InceptionV3-pool3-features Frechet
distance implementation) and `torchmetrics`'s InceptionScore for IS.

Usage:
    python benchmark.py --real_dir data/test/all --fake_dir samples/vae_eval --name VAE
    python benchmark.py --real_dir data/test/all --fake_dir samples/ddpm_eval --name DDPM
    python benchmark.py --both     # benchmark both eval dirs against real_dir in one go

Writes results to logs/benchmark_results.json and prints a summary table.
"""

import argparse
import json
import os

import conf


def parse_args():
    p = argparse.ArgumentParser(description="Compute FID + Inception Score for generated images")
    p.add_argument("--real_dir", type=str, default=os.path.join(conf.TEST_DIR, "all"))
    p.add_argument("--fake_dir", type=str, default=None)
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--both", action="store_true")
    p.add_argument("--batch_size", type=int, default=conf.FID_BATCH_SIZE)
    p.add_argument("--device", type=str, default=conf.DEVICE)
    return p.parse_args()


def compute_fid(real_dir: str, fake_dir: str, batch_size: int, device: str) -> float:
    from pytorch_fid import fid_score
    return fid_score.calculate_fid_given_paths(
        [real_dir, fake_dir], batch_size=batch_size, device=device, dims=2048,
    )


def compute_inception_score(fake_dir: str, batch_size: int, device: str, splits: int = 10):
    import torch
    from torchmetrics.image.inception import InceptionScore
    from PIL import Image
    import torchvision.transforms as T
    import glob
    from tqdm import tqdm

    files = sorted(glob.glob(os.path.join(fake_dir, "*.png")) + glob.glob(os.path.join(fake_dir, "*.jpg")))
    if not files:
        raise FileNotFoundError(f"No images found in {fake_dir}")

    transform = T.Compose([T.Resize(299), T.CenterCrop(299), T.ToTensor()])
    inception = InceptionScore(splits=splits).to(device)

    batch = []
    for f in tqdm(files, desc="Inception Score: loading+scoring"):
        img = Image.open(f).convert("RGB")
        img = transform(img)
        img = (img * 255).to(torch.uint8)
        batch.append(img)
        if len(batch) == batch_size:
            inception.update(torch.stack(batch).to(device))
            batch = []
    if batch:
        inception.update(torch.stack(batch).to(device))

    mean, std = inception.compute()
    return float(mean), float(std)


def run_benchmark(real_dir: str, fake_dir: str, name: str, batch_size: int, device: str) -> dict:
    print(f"\n=== Benchmarking '{name}' ===")
    print(f"real: {real_dir}")
    print(f"fake: {fake_dir}")

    print("\nComputing FID...")
    fid_value = compute_fid(real_dir, fake_dir, batch_size, device)
    print(f"FID: {fid_value:.4f}")

    print("\nComputing Inception Score...")
    is_mean, is_std = compute_inception_score(fake_dir, batch_size, device)
    print(f"Inception Score: {is_mean:.4f} +/- {is_std:.4f}")

    return {"name": name, "fid": fid_value, "inception_score_mean": is_mean, "inception_score_std": is_std}


def save_results(result: dict):
    os.makedirs(conf.LOG_DIR, exist_ok=True)
    results_path = os.path.join(conf.LOG_DIR, "benchmark_results.json")
    results = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
    results[result["name"]] = result
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")


def print_summary_table(results: dict):
    print("\n" + "=" * 55)
    print(f"{'Model':<12}{'FID':>12}{'IS (mean)':>16}{'IS (std)':>14}")
    print("-" * 55)
    for name, r in results.items():
        print(f"{name:<12}{r['fid']:>12.4f}{r['inception_score_mean']:>16.4f}{r['inception_score_std']:>14.4f}")
    print("=" * 55)


def main():
    args = parse_args()

    if args.both:
        vae_fake = os.path.join(conf.SAMPLE_DIR, "vae_eval")
        ddpm_fake = os.path.join(conf.SAMPLE_DIR, "ddpm_eval")
        for name, fake_dir in [("VAE", vae_fake), ("DDPM", ddpm_fake)]:
            if not os.path.isdir(fake_dir) or not os.listdir(fake_dir):
                print(f"Skipping {name}: {fake_dir} missing or empty. Run eval_vae.py / eval_ddpm.py first.")
                continue
            result = run_benchmark(args.real_dir, fake_dir, name, args.batch_size, args.device)
            save_results(result)
    else:
        if args.fake_dir is None or args.name is None:
            raise ValueError("--fake_dir and --name are required unless --both is used")
        result = run_benchmark(args.real_dir, args.fake_dir, args.name, args.batch_size, args.device)
        save_results(result)

    results_path = os.path.join(conf.LOG_DIR, "benchmark_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        print_summary_table(all_results)


if __name__ == "__main__":
    main()
