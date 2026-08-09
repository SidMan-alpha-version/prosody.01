#!/usr/bin/env python3
"""
Prosody Conformer ASR / Prosody Feature Inference Tool
"""

import argparse
import os
import torch
import torchaudio
from pathlib import Path

from model_configs import MODEL_REGISTRY
from models import ProsodyConformer, AudioProcessor, TextTokenizer


def load_model(checkpoint_path: str = None, model_size: str = "medium", device: str = "cpu"):
    """Load ProsodyConformer model from weights checkpoint or clean initialization."""
    config = MODEL_REGISTRY.get(model_size, MODEL_REGISTRY["medium"])
    model = ProsodyConformer(config)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"📦 Loading model checkpoint from: {checkpoint_path}")
        if os.path.isdir(checkpoint_path):
            bin_path = os.path.join(checkpoint_path, "pytorch_model.bin")
            pt_path = os.path.join(checkpoint_path, "model.pt")
            if os.path.exists(bin_path):
                state_dict = torch.load(bin_path, map_location=device)
            elif os.path.exists(pt_path):
                state_dict = torch.load(pt_path, map_location=device)
            else:
                state_dict = None
        else:
            state_dict = torch.load(checkpoint_path, map_location=device)

        if state_dict:
            model.load_state_dict(state_dict, strict=False)
            print("✓ Checkpoint weights loaded successfully!")

    model = model.to(device)
    model.eval()
    return model, config


def run_inference(model, audio_tensor, sample_rate=16000, device="cpu"):
    """Run Mel extraction, Conformer forward pass, and CTC decoding on audio waveform."""
    config_vocab = getattr(model, "vocab_size", 256)
    audio_processor = AudioProcessor(sample_rate=sample_rate, n_mels=80).to(device)
    tokenizer = TextTokenizer(vocab_size=config_vocab)

    if isinstance(audio_tensor, list):
        audio_tensor = torch.tensor(audio_tensor, dtype=torch.float32)

    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    audio_tensor = audio_tensor.to(device)

    with torch.no_grad():
        mel_features = audio_processor(audio_tensor)
        outputs = model(mel_features)

        ctc_logits = outputs["ctc_logits"]
        transcription = tokenizer.ctc_decode(ctc_logits[0])

        f0 = outputs["f0_pred"][0].mean().item()
        energy = outputs["energy_pred"][0].mean().item()
        duration = outputs["duration_pred"][0].mean().item()

    return {
        "transcription": transcription,
        "f0_mean": f0,
        "energy_mean": energy,
        "duration_mean": duration,
        "mel_shape": list(mel_features.shape),
    }


def main():
    parser = argparse.ArgumentParser(description="Prosody Model ASR & Feature Inference")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint directory or pytorch model file")
    parser.add_argument("--model-size", type=str, default="medium", choices=["small", "medium", "large"])
    parser.add_argument("--audio", type=str, default=None, help="Path to input audio file (.wav, .flac, .mp3)")
    parser.add_argument("--dummy-test", action="store_true", help="Run quick inference test on synthetic audio")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"⚡ Device selected for inference: {device}")

    model, config = load_model(args.checkpoint, args.model_size, device=device)

    if args.dummy_test or not args.audio:
        print("🎤 Running inference test with synthetic 16kHz audio sample...")
        sample_rate = 16000
        # Generate 1 second synthetic audio tone
        t = torch.linspace(0, 1, sample_rate)
        audio_waveform = torch.sin(2 * 3.14159 * 440 * t)  # 440Hz A tone
    else:
        print(f"🎵 Loading audio file: {args.audio}")
        audio_waveform, sample_rate = torchaudio.load(args.audio)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            audio_waveform = resampler(audio_waveform)
            sample_rate = 16000

    results = run_inference(model, audio_waveform, sample_rate=sample_rate, device=device)

    print("\n" + "=" * 60)
    print("📝 INFERENCE RESULTS")
    print("=" * 60)
    print(f"Spectrogram shape: {results['mel_shape']}")
    print(f"Transcribed Text : '{results['transcription']}'")
    print(f"Estimated F0     : {results['f0_mean']:.4f}")
    print(f"Estimated Energy : {results['energy_mean']:.4f}")
    print(f"Estimated Duration: {results['duration_mean']:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
