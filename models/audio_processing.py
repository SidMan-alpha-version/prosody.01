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

    def ctc_decode(self, logits_or_log_probs) -> str:
        """
        Greedy CTC Decoding: Takes logits [T, V] or [1, T, V],
        computes argmax over vocabulary dim, collapses adjacent repeated tokens,
        and removes CTC blank token (0).
        """
        if isinstance(logits_or_log_probs, torch.Tensor):
            if logits_or_log_probs.ndim == 3:
                logits_or_log_probs = logits_or_log_probs[0]
            argmax_tokens = torch.argmax(logits_or_log_probs, dim=-1).tolist()
        else:
            argmax_tokens = logits_or_log_probs

        # Collapse repeats and filter out blank (0)
        collapsed = []
        prev = None
        for t in argmax_tokens:
            if t != prev:
                if t != self.blank_id:
                    collapsed.append(t)
                prev = t

        return self.decode(collapsed)


def compute_levenshtein_distance(seq1, seq2):
    """Compute Levenshtein edit distance between two sequences (words or characters)."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]


def compute_wer(ref: str, hyp: str) -> float:
    """Compute Word Error Rate (WER) between reference and hypothesis text."""
    ref_words = ref.strip().split()
    hyp_words = hyp.strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    dist = compute_levenshtein_distance(ref_words, hyp_words)
    return float(dist) / len(ref_words)


def compute_cer(ref: str, hyp: str) -> float:
    """Compute Character Error Rate (CER) between reference and hypothesis text."""
    ref_chars = list(ref.strip())
    hyp_chars = list(hyp.strip())
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    dist = compute_levenshtein_distance(ref_chars, hyp_chars)
    return float(dist) / len(ref_chars)

