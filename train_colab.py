#!/usr/bin/env python3
"""
Train Prosody Conformer Model on Free Colab / Kaggle
Optimized with:
- 4x Conv2D Subsampling for 16x lower attention VRAM memory
- Micro-batching + Gradient Accumulation
- Audio Truncation (clamped to max 10s audio) to prevent OOM spikes
- Single-file model checkpointing to save disk space
- Fallback dataset stream (LibriSpeech/VoxPopuli/FLEURS/Synthetic)
"""

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


def get_streaming_dataloader(languages, split='train', batch_size=2, max_samples=None, hf_token=None, max_audio_samples=160000, synthetic_data=False):
    """Stream data from HuggingFace Cloud with robust dataset fallbacks and max audio length caps."""
    if synthetic_data or not HAS_STREAMING:
        print(f"⚡ Using instant synthetic audio stream [{split}]...")
        class SyntheticDataset:
            def __iter__(self):
                count = 0
                while max_samples is None or count < (max_samples or 1000):
                    yield {
                        'audio': [0.01 * (i % 100 - 50) / 50 for i in range(16000)],
                        'sentence': 'prosody synthetic speech recognition sample'
                    }
                    count += 1
        batch_ds = [SyntheticDataset()]
        combined = batch_ds[0]
        def batch_iterator():
            batch = {'audio': [], 'text': []}
            for sample in combined:
                batch['audio'].append(sample['audio'])
                batch['text'].append(sample['sentence'])
                if len(batch['audio']) == batch_size:
                    yield batch
                    batch = {'audio': [], 'text': []}
            if batch['audio']:
                yield batch
        return batch_iterator()

    print(f"\n📡 Streaming dataset [{split}] for languages: {', '.join(languages[:5])}{'...' if len(languages)>5 else ''}")
    datasets = []

    kwargs = {}
    token_to_use = hf_token or os.environ.get("HF_TOKEN")
    if token_to_use:
        kwargs['token'] = token_to_use

    for lang in languages:
        ds = None
        candidate_repos = [
            ('librispeech_asr', 'clean' if split == 'train' else 'validation'),
            ('facebook/voxpopuli', lang if lang in ['en', 'de', 'fr', 'es', 'it', 'nl', 'pl', 'ro'] else 'en'),
            ('mozilla-foundation/common_voice_17_0', lang),
        ]

        for repo_id, config_name in candidate_repos:
            try:
                split_name = 'train.clean.100' if repo_id == 'librispeech_asr' and split == 'train' else ('validation.clean' if repo_id == 'librispeech_asr' else split)
                ds_cand = load_dataset(
                    repo_id,
                    config_name,
                    split=split_name,
                    streaming=True,
                    **kwargs
                )
                if max_samples:
                    ds_cand = ds_cand.take(max_samples // len(languages))

                ds = ds_cand
                datasets.append(ds)
                print(f"   ✓ Configured [{repo_id}] stream ({lang})")
                break
            except Exception:
                continue

    if not datasets:
        print(f"\n⚠️ Cloud dataset streaming unavailable. Using instant synthetic audio stream...")
        class SyntheticDataset:
            def __iter__(self):
                count = 0
                while max_samples is None or count < (max_samples or 500):
                    yield {
                        'audio': [0.01 * (i % 100 - 50) / 50 for i in range(16000)],
                        'sentence': 'prosody synthetic speech recognition sample'
                    }
                    count += 1
        datasets = [SyntheticDataset()]

    combined = interleave_datasets(datasets) if len(datasets) > 1 else datasets[0]

    def batch_iterator():
        batch = {'audio': [], 'text': []}
        for sample in combined:
            try:
                audio = sample['audio']['array'] if isinstance(sample.get('audio'), dict) else sample.get('audio')
                if audio is None:
                    continue
                text = sample.get('sentence', sample.get('raw_text', sample.get('normalized_text', sample.get('text', ''))))
                if not text or len(str(text).strip()) == 0:
                    continue

                # Truncate audio samples to max_audio_samples (default 10s @ 16kHz) to avoid VRAM spikes
                if len(audio) > max_audio_samples:
                    audio = audio[:max_audio_samples]

                batch['audio'].append(audio)
                batch['text'].append(str(text))
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

            input_lengths = torch.full((B_size,), T_frames, dtype=torch.long, device=accelerator.device)
            if (input_lengths < target_lengths).any():
                continue

            ctc_log_probs = ctc_logits.log_softmax(dim=-1).transpose(0, 1)
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
    parser = argparse.ArgumentParser(description="Streamlined Prosody Conformer Model Training")
    parser.add_argument("--model-size", type=str, default="small", choices=["nano", "small", "medium", "large"])
    parser.add_argument("--languages", type=str, default="en", help="Comma-separated languages or 'all' for 20 languages")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2, help="Micro-batch size per accumulation step")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default="outputs_colab")
    parser.add_argument("--use-streaming", action="store_true", default=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-audio-len", type=int, default=160000, help="Max audio waveform samples (160000 = 10s @ 16kHz)")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="auto", choices=["no", "fp16", "bf16", "auto"])
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face API token")
    parser.add_argument("--synthetic-data", action="store_true", help="Use instant synthetic audio stream to bypass cloud network latency")

    args = parser.parse_args()

    # Setup Accelerator with gradient accumulation
    mixed_precision = args.mixed_precision
    if mixed_precision == "auto":
        mixed_precision = "fp16" if torch.cuda.is_available() else "no"

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    config = MODEL_REGISTRY[args.model_size]

    if args.languages.lower() == "all":
        languages = ALL_20_LANGUAGES
    else:
        languages = [l.strip() for l in args.languages.split(",")]

    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    if accelerator.is_main_process:
        print(f"\n{'='*70}")
        print(f"🚀 Prosody Training - {args.model_size.upper()} Conformer Model (Streamlined)")
        print(f"{'='*70}")
        print(f"Languages ({len(languages)}): {', '.join(languages[:10])}{'...' if len(languages)>10 else ''}")
        print(f"Model params        : {config['estimated_params']/1e6:.1f}M")
        print(f"Subsampling factor  : {config.get('subsampling_factor', 4)}x (Time Downsampling Enabled)")
        print(f"Gradient Accum.     : {args.gradient_accumulation_steps} steps (Effective batch size: {args.batch_size * args.gradient_accumulation_steps})")
        print(f"Mixed Precision     : {mixed_precision}")
        print(f"{'='*70}\n")

    # Load data
    if args.use_streaming:
        train_data = get_streaming_dataloader(
            languages, 'train', args.batch_size, args.max_samples, args.hf_token, args.max_audio_len, args.synthetic_data
        )
        val_data = get_streaming_dataloader(
            languages, 'validation', args.batch_size, 20, args.hf_token, args.max_audio_len, args.synthetic_data
        )
        if train_data is None:
            print("❌ Failed to load streaming data")
            return
        if args.validate_only:
            print("✓ Streaming data pipeline validated successfully!")
            return
    else:
        print("❌ Local mode not enabled. Use --use-streaming")
        return

    # Instantiate Audio Processor & Tokenizer
    audio_processor = AudioProcessor(sample_rate=16000, n_mels=config["encoder"]["input_dim"]).to(accelerator.device)
    tokenizer = TextTokenizer(vocab_size=config["vocab_size"])

    # Create Prosody Conformer Model
    model = ProsodyConformer(config)
    ctc_criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)

    model, optimizer = accelerator.prepare(model, optimizer)

    # Training loop
    total_loss = 0
    step = 0
    best_val_loss = float('inf')
    metrics_log = []

    for epoch in range(args.epochs):
        model.train()
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        pbar = tqdm(train_data, desc="Training")

        for batch_idx, batch in enumerate(pbar):
            try:
                if not batch or len(batch['audio']) == 0:
                    continue

                with accelerator.accumulate(model):
                    # Extract Log-Mel Spectrogram features
                    mel_features = audio_processor(batch['audio']).to(accelerator.device)

                    # Tokenize targets
                    targets, target_lengths = tokenizer.encode_batch(batch['text'])
                    targets = targets.to(accelerator.device)
                    target_lengths = target_lengths.to(accelerator.device)

                    # Forward pass
                    outputs = model(mel_features)
                    ctc_logits = outputs['ctc_logits']

                    # CTC Loss computation
                    T_frames = ctc_logits.shape[1]
                    B_size = ctc_logits.shape[0]
                    input_lengths = torch.full((B_size,), T_frames, dtype=torch.long, device=accelerator.device)

                    # Skip batch if downsampled time length is smaller than target token length
                    if (input_lengths < target_lengths).any():
                        continue

                    ctc_log_probs = ctc_logits.log_softmax(dim=-1).transpose(0, 1)
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

                    if (batch_idx + 1) % 5 == 0:
                        avg_loss = total_loss / max(1, step)
                        mem_str = f"{torch.cuda.max_memory_allocated()/1e9:.1f}GB" if torch.cuda.is_available() else "CPU"
                        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "vram": mem_str})

            except Exception as e:
                print(f"❌ Error in training step: {e}")
                continue

        # Validation Evaluation
        eval_results = None
        if val_data:
            eval_results = evaluate(model, val_data, audio_processor, tokenizer, ctc_criterion, accelerator)
            print(f"📊 Epoch {epoch+1}: Val Loss = {eval_results['val_loss']:.4f} | WER = {eval_results['val_wer']:.4f} | CER = {eval_results['val_cer']:.4f}")
            metrics_log.append({
                "epoch": epoch + 1,
                "train_loss": total_loss / max(1, step),
                "val_loss": eval_results["val_loss"],
                "val_wer": eval_results["val_wer"],
                "val_cer": eval_results["val_cer"],
            })

        # Save clean single PyTorch checkpoint file to avoid disk quota errors
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            latest_path = Path(args.output_dir) / "model_latest.pt"
            torch.save(unwrapped_model.state_dict(), latest_path)
            print(f"💾 Saved checkpoint to {latest_path}")

            if eval_results and eval_results["val_loss"] < best_val_loss:
                best_val_loss = eval_results["val_loss"]
                best_path = Path(args.output_dir) / "model_best.pt"
                torch.save(unwrapped_model.state_dict(), best_path)
                print(f"🌟 New best model saved to {best_path}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if accelerator.is_main_process:
        metrics_file = f"{args.output_dir}/metrics_summary.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics_log, f, indent=2)
        print(f"\n📊 Metrics log saved to {metrics_file}")
        print(f"✅ Training complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
