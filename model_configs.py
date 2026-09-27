#!/usr/bin/env python3
"""
Prosody ASR/TTS Model Configuration

Pre-defined model sizes optimized for free Colab (15GB GPU memory)
"""

# PROSODY_NANO: ~25M parameters - Ultra-fast streamlined tier for quick training
PROSODY_NANO = {
    "name": "prosody_nano",
    "encoder": {
        "input_dim": 80,
        "hidden_dim": 256,
        "num_layers": 8,
        "num_heads": 4,
        "ffn_dim": 1024,
        "conv_kernel_size": 31,
        "dropout": 0.1,
    },
    "decoder": {
        "hidden_dim": 256,
        "num_layers": 4,
        "num_heads": 4,
        "ffn_dim": 1024,
        "dropout": 0.1,
    },
    "vocab_size": 256,
    "subsampling": True,
    "subsampling_factor": 4,
    "gradient_checkpointing": True,
    "estimated_params": 25_000_000,
    "estimated_memory_gb": 0.6,
}

# PROSODY_SMALL: 85M parameters - Streamlined for free Colab/Kaggle T4
PROSODY_SMALL = {
    "name": "prosody_small",
    "encoder": {
        "input_dim": 80,
        "hidden_dim": 512,
        "num_layers": 12,
        "num_heads": 8,
        "ffn_dim": 2048,
        "conv_kernel_size": 31,
        "dropout": 0.1,
    },
    "decoder": {
        "hidden_dim": 512,
        "num_layers": 6,
        "num_heads": 8,
        "ffn_dim": 2048,
        "dropout": 0.1,
    },
    "vocab_size": 256,
    "subsampling": True,
    "subsampling_factor": 4,
    "gradient_checkpointing": True,
    "estimated_params": 85_000_000,
    "estimated_memory_gb": 1.2,
}

# PROSODY_MEDIUM: 274M parameters - High performance with FP16 + Subsampling
PROSODY_MEDIUM = {
    "name": "prosody_medium",
    "encoder": {
        "input_dim": 80,  # mel-spectrogram bins
        "hidden_dim": 768,
        "num_layers": 18,
        "num_heads": 12,
        "ffn_dim": 3072,
        "conv_kernel_size": 31,
        "dropout": 0.1,
    },
    "decoder": {
        "hidden_dim": 768,
        "num_layers": 8,
        "num_heads": 12,
        "ffn_dim": 3072,
        "dropout": 0.1,
    },
    "vocab_size": 256,
    "subsampling": True,
    "subsampling_factor": 4,
    "gradient_checkpointing": True,
    "estimated_params": 274_000_000,
    "estimated_memory_gb": 3.1,
}

PROSODY_LARGE = {
    "name": "prosody_large",
    "encoder": {
        "input_dim": 80,
        "hidden_dim": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "ffn_dim": 4096,
        "conv_kernel_size": 31,
        "dropout": 0.1,
    },
    "decoder": {
        "hidden_dim": 1024,
        "num_layers": 12,
        "num_heads": 16,
        "ffn_dim": 4096,
        "dropout": 0.1,
    },
    "vocab_size": 256,
    "subsampling": True,
    "subsampling_factor": 4,
    "gradient_checkpointing": True,
    "estimated_params": 662_000_000,
    "estimated_memory_gb": 8.5,
}

MODEL_REGISTRY = {
    "nano": PROSODY_NANO,
    "small": PROSODY_SMALL,
    "medium": PROSODY_MEDIUM,
    "large": PROSODY_LARGE,
}

def get_config(size: str = "medium"):
    """Get model config by size."""
    if size not in MODEL_REGISTRY:
        raise ValueError(f"Unknown size: {size}. Choose: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[size]
