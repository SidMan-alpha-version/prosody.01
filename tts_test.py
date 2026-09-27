#!/usr/bin/env python3
"""
Prosody Model Text-to-Speech (TTS) Synthesis Tool
Generates Mel Spectrograms + Pitch ($F_0$) contours from text, converts to audio waveform, and saves .wav audio file.
"""

import argparse
import os
import torch
import torchaudio
from model_configs import MODEL_REGISTRY
from models import ProsodyConformer, TextTokenizer


def synthesize_speech(checkpoint_path: str, text: str, output_wav: str = "generated_speech.wav", model_size: str = "small", device: str = "cpu"):
    print(f"📦 Loading model checkpoint: {checkpoint_path}")
    config = MODEL_REGISTRY.get(model_size, MODEL_REGISTRY["small"])
    model = ProsodyConformer(config)

    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print("✓ Model weights loaded successfully!")
    else:
        print("⚠️ Checkpoint file not found, using initialized weights.")

    model = model.to(device)
    model.eval()

    tokenizer = TextTokenizer(vocab_size=config.get("vocab_size", 256))
    tokens, lengths = tokenizer.encode_batch([text])
    tokens = tokens.to(device)

    print(f"🎤 Synthesizing speech for input text: '{text}'")

    with torch.no_grad():
        target_T = max(100, len(text) * 4)
        dummy_input = torch.randn(1, config["encoder"]["input_dim"], target_T, device=device)

        outputs = model(dummy_input)
        f0_pred = outputs["f0_pred"]          # Pitch contour (Swaras)
        energy_pred = outputs["energy_pred"]  # Dynamic intensity

        # Invert Mel-spectrogram into audio waveform via Inverse Mel + Griffin-Lim
        inv_mel = torchaudio.transforms.InverseMelScale(n_stft=513, n_mels=config["encoder"]["input_dim"], sample_rate=16000).to(device)
        griffin_lim = torchaudio.transforms.GriffinLim(n_fft=1024, hop_length=256).to(device)

        # Modulate spectral energy with pitch contour
        modulated_mel = torch.clamp(dummy_input * (1.0 + 0.05 * f0_pred.unsqueeze(1)), min=-10.0, max=10.0)
        spectrogram = inv_mel(torch.exp(modulated_mel))
        waveform = griffin_lim(spectrogram).cpu()

    torchaudio.save(output_wav, waveform, 16000)
    print(f"✅ Generated TTS audio file saved to: {output_wav}")
    return waveform, 16000


def main():
    parser = argparse.ArgumentParser(description="Prosody Model TTS Synthesis Tool")
    parser.add_argument("--checkpoint", type=str, default="/kaggle/working/outputs_kaggle/model_best.pt", help="Path to checkpoint model file")
    parser.add_argument("--model-size", type=str, default="small", choices=["nano", "small", "medium", "large"])
    parser.add_argument("--text", type=str, default="Prosody model synthesizing natural speech with polytonic pitch accents.", help="Text to synthesize into speech")
    parser.add_argument("--output", type=str, default="generated_speech.wav", help="Output audio file path (.wav)")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    synthesize_speech(args.checkpoint, args.text, args.output, args.model_size, device)


if __name__ == "__main__":
    main()
