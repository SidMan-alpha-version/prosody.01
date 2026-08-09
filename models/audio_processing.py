#!/usr/bin/env python3
"""
Audio Feature Extraction and Text Tokenization for Prosody Model
"""

import torch
import torch.nn as nn
import torchaudio.transforms as T


class AudioProcessor(nn.Module):
    """Converts raw audio waveforms into 80-channel log-mel spectrogram features."""

    def __init__(self, sample_rate=16000, n_mels=80, n_fft=1024, hop_length=256):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )

    def forward(self, audio_tensors):
        """
        Input: list of 1D audio waveform tensors or 2D batch tensor [B, T_raw]
        Output: mel features tensor [B, n_mels, T_frames]
        """
        if isinstance(audio_tensors, list):
            # Pad audio sequences to max length in batch
            max_len = max(len(a) for a in audio_tensors)
            padded = []
            for a in audio_tensors:
                if not isinstance(a, torch.Tensor):
                    a = torch.tensor(a, dtype=torch.float32)
                if len(a) < max_len:
                    a = torch.nn.functional.pad(a, (0, max_len - len(a)))
                padded.append(a[:max_len])
            audio_tensors = torch.stack(padded)

        if audio_tensors.ndim == 1:
            audio_tensors = audio_tensors.unsqueeze(0)

        # Move mel_spectrogram to match audio_tensors device
        if hasattr(self.mel_spectrogram, "spectrogram") and hasattr(self.mel_spectrogram.spectrogram, "window"):
            if self.mel_spectrogram.spectrogram.window is not None and audio_tensors.device != self.mel_spectrogram.spectrogram.window.device:
                self.mel_spectrogram = self.mel_spectrogram.to(audio_tensors.device)

        # Compute mel spectrogram
        mel_spec = self.mel_spectrogram(audio_tensors)  # [B, n_mels, T_frames]
        log_mel = torch.log(torch.clamp(mel_spec, min=1e-5))
        return log_mel


class TextTokenizer:
    """UTF-8 Character / Byte-level tokenizer (vocab_size = 256)."""

    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        self.blank_id = 0

    def encode(self, text: str) -> torch.Tensor:
        """Convert string to byte token IDs (1-based to reserve 0 for CTC blank)."""
        bytes_data = text.encode("utf-8", errors="ignore")
        # Shift byte values by 1 to reserve token 0 for CTC blank token
        token_ids = [(b % (self.vocab_size - 1)) + 1 for b in bytes_data]
        if not token_ids:
            token_ids = [1]
        return torch.tensor(token_ids, dtype=torch.long)

    def encode_batch(self, text_list):
        """Encode list of strings into padded tensor [B, max_target_len] and lengths tensor."""
        encoded = [self.encode(t) for t in text_list]
        lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
        padded = torch.nn.utils.rnn.pad_sequence(encoded, batch_first=True, padding_value=self.blank_id)
        return padded, lengths

    def decode(self, token_ids) -> str:
        """Decode sequence of token IDs back to UTF-8 text string."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        clean_ids = [max(0, (t - 1)) for t in token_ids if t > 0]
        return bytes(clean_ids).decode("utf-8", errors="ignore")
