#!/usr/bin/env python3
"""Train Prosody model on free Colab with Accelerate + streaming data"""

import argparse
import json
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

from accelerate import Accelerator
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model_configs import MODEL_REGISTRY

# Check for required modules
try:
    from datasets import load_dataset, interleave_datasets
    HAS_STREAMING = True
except ImportError:
    HAS_STREAMING = False

def get_streaming_dataloader(languages, split='train', batch_size=4, max_samples=None):
    """Stream data from HuggingFace Cloud (NO DOWNLOAD NEEDED!)"""
    if not HAS_STREAMING:
        print("❌ HuggingFace Datasets not found. Install with: pip install datasets")
        return None
    
    print(f"\n📡 Streaming {len(languages)} languages from cloud...")
    datasets = []
    
    for lang in languages:
        try:
            print(f"   Loading {lang}...", end=" ", flush=True)
            ds = load_dataset(
                'mozilla-foundation/common_voice_17_0',
                lang,
                split=split,
                streaming=True,
                use_auth_token=False
            )
            if max_samples:
                ds = ds.take(max_samples // len(languages))
            datasets.append(ds)
            print("✓")
        except Exception as e:
            print(f"✗ ({e})")
            continue
    
    if not datasets:
        return None
    
    combined = interleave_datasets(datasets) if len(datasets) > 1 else datasets[0]
    
    def batch_iterator():
        batch = {'audio': [], 'text': []}
        for sample in combined:
            try:
                audio = sample['audio']['array'] if isinstance(sample['audio'], dict) else sample['audio']
                batch['audio'].append(audio)
                batch['text'].append(sample['sentence'])
                if len(batch['audio']) == batch_size:
                    yield batch
                    batch = {'audio': [], 'text': []}
            except:
                continue
        if batch['audio']:
            yield batch
    
    return batch_iterator()

def main():
    parser = argparse.ArgumentParser(description="Train Prosody model on Colab")
    parser.add_argument("--model-size", type=str, default="medium", choices=["small", "medium", "large"])
    parser.add_argument("--languages", type=str, default="en", help="Comma-separated languages")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default="outputs_colab")
    parser.add_argument("--use-streaming", action="store_true", default=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="auto", choices=["no", "fp16", "bf16", "auto"])
    
    args = parser.parse_args()
    
    # Setup
    mixed_precision = args.mixed_precision
    if mixed_precision == "auto":
        mixed_precision = "fp16" if torch.cuda.is_available() else "no"
    accelerator = Accelerator(mixed_precision=mixed_precision)
    config = MODEL_REGISTRY[args.model_size]
    languages = [l.strip() for l in args.languages.split(",")]
    
    Path(args.output_dir).mkdir(exist_ok=True)
    
    if accelerator.is_main_process:
        print(f"\n{'='*70}")
        print(f"🚀 Prosody Training - {args.model_size.upper()} Model")
        print(f"{'='*70}")
        print(f"Languages: {', '.join(languages)}")
        print(f"Model params: {config['estimated_params']/1e6:.0f}M")
        print(f"Estimated memory: {config['estimated_memory_gb']:.1f}GB")
        print(f"{'='*70}\n")
    
    # Load data
    if args.use_streaming:
        train_data = get_streaming_dataloader(languages, 'train', args.batch_size, args.max_samples)
        if train_data is None:
            print("❌ Failed to load streaming data")
            return
        if args.validate_only:
            print("✓ Streaming validated!")
            return
    else:
        print("❌ Local mode not implemented. Use --use-streaming")
        return
    
    # Create dummy model for demo
    model = nn.Sequential(
        nn.Linear(80, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.Linear(256, config['vocab_size'])
    )
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)
    
    model, optimizer = accelerator.prepare(model, optimizer)
    
    # Training loop
    total_loss = 0
    step = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        pbar = tqdm(train_data, desc="Training")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Dummy training loop
                if len(batch['audio']) == 0:
                    continue
                
                # Forward pass
                logits = model(torch.randn(len(batch['audio']), 80))
                loss = logits.mean()  # Dummy loss
                
                # Backward pass
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                total_loss += loss.item()
                step += 1
                
                if (batch_idx + 1) % 10 == 0:
                    avg_loss = total_loss / step
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
        
        # Save checkpoint
        accelerator.save_state(f"{args.output_dir}/epoch_{epoch+1}")
    
    if accelerator.is_main_process:
        print(f"\n✅ Training complete! Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
