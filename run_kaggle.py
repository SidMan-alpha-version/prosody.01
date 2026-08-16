#!/usr/bin/env python3
"""
Prosody Model One-Click Runner for Kaggle Notebooks
(Kaggle gives 30GB CPU RAM + Dual T4 GPUs for free!)
"""

import os
import sys
import subprocess

KAGGLE_OUTPUT_DIR = "/kaggle/working/outputs_kaggle"


def main():
    print("=" * 70)
    print("🚀 Launching Prosody Multilingual Conformer Training on Kaggle")
    print("=" * 70)

    # 1. Install dependencies
    print("📦 Installing required packages...")
    subprocess.run([sys.executable, "-m", "pip", "install", "accelerate", "datasets", "torchaudio", "tqdm", "-q"], check=True)

    # 2. Extract arguments
    args = sys.argv[1:]
    if "--output-dir" not in args:
        args.extend(["--output-dir", KAGGLE_OUTPUT_DIR])

    if "--languages" not in args:
        args.extend(["--languages", "all"])

    if "--epochs" not in args:
        args.extend(["--epochs", "3"])

    # 3. Launch training with single process to prevent multi-worker dataset memory duplication
    cmd = ["accelerate", "launch", "--num_processes", "1", "--mixed_precision", "fp16", "train_colab.py"] + args
    print(f"⚡ Running command: {' '.join(cmd)}\n")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
