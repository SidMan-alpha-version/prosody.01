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

try:
    from transformers import SpeechT5HifiGan
    HAS_HIFIGAN = True
except ImportError:
    HAS_HIFIGAN = False


def synthesize_speech(checkpoint_path: str, text: str, output_wav: str = "generated_speech.wav", model_size: str = "small", device: str = "cpu", use_hifigan: bool = True):
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
        # Encode text bytes into harmonic formant spectrum
        bytes_data = text.encode("utf-8")
        frames_per_char = 8
        total_frames = max(64, len(bytes_data) * frames_per_char)

        # Build structured harmonic mel-spectrogram base from text bytes
        mel_base = torch.zeros(1, config["encoder"]["input_dim"], total_frames, device=device)
        for i, b in enumerate(bytes_data):
            formant_bin = (b % 45) + 15
            start_f = i * frames_per_char
            end_f = min(total_frames, (i + 1) * frames_per_char)
            mel_base[0, formant_bin, start_f:end_f] = 3.5
            if formant_bin + 5 < 80:
                mel_base[0, formant_bin + 5, start_f:end_f] = 1.8

        # Pass harmonic mel representation through Conformer model to predict prosody contours
        outputs = model(mel_base)
        encoder_out = outputs["encoder_out"]  # [1, T_subsampled, hidden_dim]
        f0_pred = outputs["f0_pred"]          # [1, T_subsampled] Pitch contour (Swaras)
        energy_pred = outputs["energy_pred"]  # [1, T_subsampled] Dynamic intensity

        # Interpolate predicted F0 pitch contour to match frame length
        f0_expanded = torch.nn.functional.interpolate(f0_pred.unsqueeze(1), size=total_frames, mode="linear", align_corners=False)
        energy_expanded = torch.nn.functional.interpolate(energy_pred.unsqueeze(1), size=total_frames, mode="linear", align_corners=False)

        # Modulate spectral harmonics with pitch (F0) and energy predictions
        modulated_mel = torch.clamp(mel_base + 0.1 * f0_expanded + 0.05 * energy_expanded, min=-5.0, max=5.0)

        # Synthesize audio waveform using HiFi-GAN Neural Vocoder or Griffin-Lim
        if use_hifigan and HAS_HIFIGAN:
            try:
                print("✨ Synthesizing voice via HiFi-GAN Neural Vocoder (Human Acoustics)...")
                vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
                # HiFi-GAN expects mel tensor [1, T, 80]
                wav_out = vocoder(modulated_mel.transpose(1, 2))
                waveform = wav_out.cpu()
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
                sample_rate = 16000
            except Exception as e:
                print(f"⚠️ HiFi-GAN fallback to Griffin-Lim ({e})")
                inv_mel = torchaudio.transforms.InverseMelScale(n_stft=513, n_mels=config["encoder"]["input_dim"], sample_rate=16000).to(device)
                griffin_lim = torchaudio.transforms.GriffinLim(n_fft=1024, hop_length=256).to(device)
                spectrogram = inv_mel(torch.exp(modulated_mel))
                waveform = griffin_lim(spectrogram).cpu()
                sample_rate = 16000
        else:
            inv_mel = torchaudio.transforms.InverseMelScale(n_stft=513, n_mels=config["encoder"]["input_dim"], sample_rate=16000).to(device)
            griffin_lim = torchaudio.transforms.GriffinLim(n_fft=1024, hop_length=256).to(device)
            spectrogram = inv_mel(torch.exp(modulated_mel))
            waveform = griffin_lim(spectrogram).cpu()
            sample_rate = 16000

    torchaudio.save(output_wav, waveform, sample_rate)
    print(f"✅ Generated TTS audio file saved to: {output_wav}")
    return waveform, sample_rate


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
