#!/usr/bin/env python3
"""
PyTorch Conformer Architecture for Prosody ASR/TTS
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    """Swish (SiLU) Activation Function."""
    def forward(self, x):
        return x * torch.sigmoid(x)


class FeedForwardModule(nn.Module):
    """Macaron-style Feed-Forward Module."""

    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.w_1 = nn.Linear(d_model, ffn_dim)
        self.act = Swish()
        self.dropout_1 = nn.Dropout(dropout)
        self.w_2 = nn.Linear(ffn_dim, d_model)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout_1(x)
        x = self.w_2(x)
        x = self.dropout_2(x)
        return residual + 0.5 * x


class ConformerConvModule(nn.Module):
    """Conformer Convolution Module with Depthwise Separable Conv1D."""

    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size must be odd"
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.act = Swish()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input shape: [B, T, C]
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # [B, C, T]

        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.act(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)

        x = x.transpose(1, 2)  # [B, T, C]
        return residual + x


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention Module."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        attn_out, _ = self.attn(x, x, x)
        x = self.dropout(attn_out)
        return residual + x


class ConformerBlock(nn.Module):
    """Full Conformer Block combining FFN, Attention, Conv, FFN, and LayerNorm."""

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, conv_kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, ffn_dim, dropout)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.conv = ConformerConvModule(d_model, conv_kernel_size, dropout)
        self.ffn2 = FeedForwardModule(d_model, ffn_dim, dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.attn(x)
        x = self.conv(x)
        x = self.ffn2(x)
        x = self.layer_norm(x)
        return x


class ConformerEncoder(nn.Module):
    """Stacked Conformer Encoder with subsampling / input projection."""

    def __init__(self, input_dim: int = 80, hidden_dim: int = 768, num_layers: int = 18, num_heads: int = 12, ffn_dim: int = 3072, conv_kernel_size: int = 31, dropout: float = 0.1, gradient_checkpointing: bool = False):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            Swish(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList([
            ConformerBlock(hidden_dim, num_heads, ffn_dim, conv_kernel_size, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, mel_features):
        """
        Input: [B, n_mels, T]
        Output: [B, T, hidden_dim]
        """
        x = mel_features.transpose(1, 2)  # [B, T, n_mels]
        x = self.input_projection(x)       # [B, T, hidden_dim]
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x


class ProsodyConformer(nn.Module):
    """
    Complete Prosody Conformer Model with:
    - Conformer Encoder
    - CTC Head for ASR/Transcription (vocab_size=256)
    - Prosody Feature Heads (F0 Pitch, Energy, Duration Predictor)
    """

    def __init__(self, config: dict):
        super().__init__()
        enc_cfg = config["encoder"]
        self.vocab_size = config.get("vocab_size", 256)
        grad_chk = config.get("gradient_checkpointing", False)

        self.encoder = ConformerEncoder(
            input_dim=enc_cfg.get("input_dim", 80),
            hidden_dim=enc_cfg.get("hidden_dim", 768),
            num_layers=enc_cfg.get("num_layers", 18),
            num_heads=enc_cfg.get("num_heads", 12),
            ffn_dim=enc_cfg.get("ffn_dim", 3072),
            conv_kernel_size=enc_cfg.get("conv_kernel_size", 31),
            dropout=enc_cfg.get("dropout", 0.1),
            gradient_checkpointing=grad_chk,
        )

        hidden_dim = enc_cfg.get("hidden_dim", 768)

        # CTC ASR Prediction Head
        self.ctc_head = nn.Linear(hidden_dim, self.vocab_size)

        # Prosody Heads (F0 pitch, energy, duration estimation)
        self.f0_head = nn.Linear(hidden_dim, 1)
        self.energy_head = nn.Linear(hidden_dim, 1)
        self.duration_head = nn.Linear(hidden_dim, 1)

    def forward(self, mel_features):
        """
        Input: mel_features [B, n_mels, T]
        Output: dict containing ctc_logits, f0_pred, energy_pred, duration_pred
        """
        encoder_out = self.encoder(mel_features)  # [B, T, hidden_dim]

        ctc_logits = self.ctc_head(encoder_out)          # [B, T, vocab_size]
        f0_pred = self.f0_head(encoder_out).squeeze(-1)    # [B, T]
        energy_pred = self.energy_head(encoder_out).squeeze(-1)  # [B, T]
        duration_pred = self.duration_head(encoder_out).squeeze(-1)  # [B, T]

        return {
            "ctc_logits": ctc_logits,
            "f0_pred": f0_pred,
            "energy_pred": energy_pred,
            "duration_pred": duration_pred,
            "encoder_out": encoder_out,
        }
