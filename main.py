"""
Usage:
    python main.py                    # full pipeline, default conf.py settings
    python main.py --skip_data        # data/ already prepared, skip re-download
    python main.py --vae_only
    python main.py --ddpm_only
"""

import argparse
import subprocess
import sys


def run(cmd: list):
    print(f"\n$ {' '.join(cmd)}\n")
    result = subprocess.run([sys.executable] + cmd)
    if result.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}")
        sys.exit(result.returncode)


def parse_args():
    p = argparse.ArgumentParser(description="Run the full anime-face VAE+DDPM pipeline locally")
    p.add_argument("--skip_data", action="store_true")
    p.add_argument("--vae_only", action="store_true")
    p.add_argument("--ddpm_only", action="store_true")
    p.add_argument("--skip_benchmark", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    do_vae = not args.ddpm_only
    do_ddpm = not args.vae_only

    if not args.skip_data:
        run(["data.py"])

    if do_vae:
        run(["train_vae.py"])
        run(["eval_vae.py"])

    if do_ddpm:
        run(["train_ddpm.py"])
        run(["eval_ddpm.py"])

    if not args.skip_benchmark:
        run(["benchmark.py", "--both"])

    print("\nPipeline complete. Checkpoints in checkpoints/, samples in samples/, logs in logs/.")


if __name__ == "__main__":
    main()
