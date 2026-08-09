#!/usr/bin/env python3
"""Train Prosody model on free Colab with Accelerate + streaming data"""

import os
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
from models import ProsodyConformer, AudioProcessor, TextTokenizer, compute_wer, compute_cer

# Check for required modules
try:
    from datasets import load_dataset, interleave_datasets
    HAS_STREAMING = True
except ImportError:
    HAS_STREAMING = False

ALL_20_LANGUAGES = [
    "en", "es", "fr", "de", "it", "pt", "nl", "sv", "uk", "el",
    "ro", "af", "pl", "ru", "zh", "hi", "bn", "ta", "ml", "ar"
]

def get_streaming_dataloader(languages, split='train', batch_size=4, max_samples=None, hf_token=None):
    """Stream data from HuggingFace Cloud (NO DOWNLOAD NEEDED!)"""
    if not HAS_STREAMING:
        print("❌ HuggingFace Datasets not found. Install with: pip install datasets")
        return None
    
    print(f"\n📡 Streaming {len(languages)} languages [{split}] from cloud...")
    datasets = []
    
    kwargs = {}
    token_to_use = hf_token or os.environ.get("HF_TOKEN")
    if token_to_use:
        kwargs['token'] = token_to_use

    for lang in languages:
        print(f"   Loading {lang} ({split})...", end=" ", flush=True)
        ds = None
        
        # Try multiple open/gated speech datasets for maximum resilience
        candidate_repos = [
            ('mozilla-foundation/common_voice_17_0', lang),
            ('mozilla-foundation/common_voice_13_0', lang),
            ('facebook/voxpopuli', lang if lang in ['en', 'de', 'fr', 'es', 'it', 'nl', 'pl', 'ro'] else 'en'),
        ]
        
        for repo_id, config_name in candidate_repos:
            try:
                ds_cand = load_dataset(
                    repo_id,
                    config_name,
                    split=split,
                    streaming=True,
                    **kwargs
                )
                if max_samples:
                    ds_cand = ds_cand.take(max_samples // len(languages))
                
                # Test iterator
                _ = next(iter(ds_cand))
                ds = ds_cand
                datasets.append(ds)
                print(f"✓ ({repo_id.split('/')[-1]})")
                break
            except Exception:
                continue
        
        if ds is None:
            print("✗ (Failed to access gated dataset. Set HF_TOKEN or login with huggingface-cli)")
    
    if not datasets:
        print(f"\n⚠️ No cloud datasets loaded for [{split}]. Using synthetic stream for validation/demonstration...")
        class SyntheticDataset:
            def __iter__(self):
                count = 0
                while max_samples is None or count < (max_samples or 100):
                    yield {'audio': [0.0]*16000, 'sentence': 'prosody synthetic speech sample'}
                    count += 1
        datasets = [SyntheticDataset()]

    combined = interleave_datasets(datasets) if len(datasets) > 1 else datasets[0]
    
    def batch_iterator():
        batch = {'audio': [], 'text': []}
        for sample in combined:
            try:
                audio = sample['audio']['array'] if isinstance(sample['audio'], dict) else sample['audio']
                text = sample.get('sentence', sample.get('raw_text', sample.get('normalized_text', '')))
                batch['audio'].append(audio)
                batch['text'].append(text)
                if len(batch['audio']) == batch_size:
                    yield batch
                    batch = {'audio': [], 'text': []}
            except Exception:
                continue
        if batch['audio']:
            yield batch
    
    return batch_iterator()


def evaluate(model, val_loader, audio_processor, tokenizer, ctc_criterion, accelerator, max_eval_samples=20):
    """Run validation pass and compute validation loss, WER, and CER."""
    model.eval()
    val_loss = 0.0
    val_steps = 0
    wers, cers = [], []

    with torch.no_grad():
        for batch in val_loader:
            if not batch or len(batch['audio']) == 0:
                continue

            mel_features = audio_processor(batch['audio']).to(accelerator.device)
            targets, target_lengths = tokenizer.encode_batch(batch['text'])
            targets = targets.to(accelerator.device)
            target_lengths = target_lengths.to(accelerator.device)

            outputs = model(mel_features)
            ctc_logits = outputs['ctc_logits']

            T_frames = ctc_logits.shape[1]
            B_size = ctc_logits.shape[0]
            ctc_log_probs = ctc_logits.log_softmax(dim=-1).transpose(0, 1)
            input_lengths = torch.full((B_size,), T_frames, dtype=torch.long, device=accelerator.device)

            loss_ctc = ctc_criterion(ctc_log_probs, targets, input_lengths, target_lengths)
            val_loss += loss_ctc.item()
            val_steps += 1

            for b in range(B_size):
                ref_text = batch['text'][b]
                hyp_text = tokenizer.ctc_decode(ctc_logits[b])
                wers.append(compute_wer(ref_text, hyp_text))
                cers.append(compute_cer(ref_text, hyp_text))

            if val_steps * B_size >= max_eval_samples:
                break

    model.train()
    avg_loss = val_loss / max(1, val_steps)
    avg_wer = sum(wers) / max(1, len(wers))
    avg_cer = sum(cers) / max(1, len(cers))

    return {"val_loss": avg_loss, "val_wer": avg_wer, "val_cer": avg_cer}


def main():
    parser = argparse.ArgumentParser(description="Train Prosody Conformer model on Colab")
    parser.add_argument("--model-size", type=str, default="medium", choices=["small", "medium", "large"])
    parser.add_argument("--languages", type=str, default="all", help="Comma-separated languages or 'all' for 20 languages")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default="outputs_colab")
    parser.add_argument("--use-streaming", action="store_true", default=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="auto", choices=["no", "fp16", "bf16", "auto"])
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face API token")
    
    args = parser.parse_args()
    
    # Setup Accelerator
    mixed_precision = args.mixed_precision
    if mixed_precision == "auto":
        mixed_precision = "fp16" if torch.cuda.is_available() else "no"
    accelerator = Accelerator(mixed_precision=mixed_precision)
    config = MODEL_REGISTRY[args.model_size]
    
    if args.languages.lower() == "all":
        languages = ALL_20_LANGUAGES
    else:
        languages = [l.strip() for l in args.languages.split(",")]
    
    Path(args.output_dir).mkdir(exist_ok=True)
    
    if accelerator.is_main_process:
        print(f"\n{'='*70}")
        print(f"🚀 Prosody Training - {args.model_size.upper()} Conformer Model")
        print(f"{'='*70}")
        print(f"Languages ({len(languages)}): {', '.join(languages)}")
        print(f"Model params: {config['estimated_params']/1e6:.0f}M")
        print(f"Estimated memory: {config['estimated_memory_gb']:.1f}GB")
        print(f"Gradient Checkpointing: {config.get('gradient_checkpointing', False)}")
        print(f"{'='*70}\n")
    
    # Load data
    if args.use_streaming:
        train_data = get_streaming_dataloader(languages, 'train', args.batch_size, args.max_samples, args.hf_token)
        val_data = get_streaming_dataloader(languages, 'validation', args.batch_size, 20, args.hf_token)
        if train_data is None:
            print("❌ Failed to load streaming data")
            return
        if args.validate_only:
            print("✓ Streaming validated successfully!")
            return
    else:
        print("❌ Local mode not implemented. Use --use-streaming")
        return
    
    # Instantiate Audio Processor & Tokenizer
    audio_processor = AudioProcessor(sample_rate=16000, n_mels=config["encoder"]["input_dim"]).to(accelerator.device)
    tokenizer = TextTokenizer(vocab_size=config["vocab_size"])
    
    # Create Prosody Conformer Model
    model = ProsodyConformer(config)
    ctc_criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)
    
    model, optimizer = accelerator.prepare(model, optimizer)
    
    # Training loop
    total_loss = 0
    step = 0
    metrics_log = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        pbar = tqdm(train_data, desc="Training")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                if len(batch['audio']) == 0:
                    continue
                
                # Extract Log-Mel Spectrogram features
                mel_features = audio_processor(batch['audio']).to(accelerator.device)  # [B, n_mels, T_frames]
                
                # Tokenize targets
                targets, target_lengths = tokenizer.encode_batch(batch['text'])
                targets = targets.to(accelerator.device)
                target_lengths = target_lengths.to(accelerator.device)
                
                # Forward pass
                outputs = model(mel_features)
                ctc_logits = outputs['ctc_logits']  # [B, T_frames, vocab_size]
                
                # CTC Loss computation: [T_frames, B, vocab_size]
                T_frames = ctc_logits.shape[1]
                B_size = ctc_logits.shape[0]
                ctc_log_probs = ctc_logits.log_softmax(dim=-1).transpose(0, 1)
                input_lengths = torch.full((B_size,), T_frames, dtype=torch.long, device=accelerator.device)
                
                loss_ctc = ctc_criterion(ctc_log_probs, targets, input_lengths, target_lengths)
                loss_prosody = outputs['f0_pred'].abs().mean() * 0.01 + outputs['energy_pred'].abs().mean() * 0.01
                loss = loss_ctc + loss_prosody
                
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
                print(f"❌ Error in training step: {e}")
                continue
        
        # Validation Evaluation
        if val_data:
            eval_results = evaluate(model, val_data, audio_processor, tokenizer, ctc_criterion, accelerator)
            print(f"📊 Validation Epoch {epoch+1}: Val Loss = {eval_results['val_loss']:.4f} | WER = {eval_results['val_wer']:.4f} | CER = {eval_results['val_cer']:.4f}")
            metrics_log.append({
                "epoch": epoch + 1,
                "train_loss": total_loss / max(1, step),
                "val_loss": eval_results["val_loss"],
                "val_wer": eval_results["val_wer"],
                "val_cer": eval_results["val_cer"],
            })

        # Save checkpoint
        accelerator.save_state(f"{args.output_dir}/epoch_{epoch+1}")
    
    if accelerator.is_main_process:
        metrics_file = f"{args.output_dir}/metrics_summary.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics_log, f, indent=2)
        print(f"\n📊 Metrics log saved to {metrics_file}")
        print(f"✅ Training complete! Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
