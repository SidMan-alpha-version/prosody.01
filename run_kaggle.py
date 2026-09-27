#!/usr/bin/env python3
"""
Prosody Model One-Click Runner for Kaggle Notebooks & Colab
(Optimized for zero-crash execution, minimal memory, and low disk usage)
"""

import os
import sys
import subprocess

KAGGLE_OUTPUT_DIR = "/kaggle/working/outputs_kaggle"


def main():
    print("=" * 70)
    print("🚀 Launching Streamlined Prosody Conformer Training")
    print("=" * 70)

    # 1. Install dependencies
    print("📦 Installing required packages...")
    subprocess.run([sys.executable, "-m", "pip", "install", "accelerate", "datasets", "torchaudio", "soundfile", "librosa", "tqdm", "transformers", "huggingface_hub", "-q"], check=False)

    # 2. Extract arguments
    args = sys.argv[1:]
    if "--output-dir" not in args:
        args.extend(["--output-dir", KAGGLE_OUTPUT_DIR])

    if "--model-size" not in args:
        args.extend(["--model-size", "small"])

    if "--batch-size" not in args:
        args.extend(["--batch-size", "2"])

    if "--gradient-accumulation-steps" not in args:
        args.extend(["--gradient-accumulation-steps", "4"])

    if "--epochs" not in args:
        args.extend(["--epochs", "3"])

    # 3. Launch training script
    cmd = [sys.executable, "train_colab.py"] + args
    print(f"⚡ Running training command: {' '.join(cmd)}\n")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
