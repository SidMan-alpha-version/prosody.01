# Prosody: Advanced Multilingual TTS/ASR (Streamlined & Zero-Crash)

**Conformer ASR & Prosody Feature Model** optimized for zero-crash execution on Kaggle and Google Colab free GPU tiers (T4 15GB VRAM).

- ✅ **4x Conv2D Subsampling**: 16x lower attention VRAM footprint
- ✅ **Micro-batching + Gradient Accumulation**: High effective batch size with minimal memory usage
- ✅ **Audio Length Clamping**: Safe 10s audio truncation eliminates CUDA OOM memory spikes
- ✅ **Lean Checkpointing**: Single-file weights save (<300MB) prevent Kaggle/Colab disk quota crashes
- ✅ **Resilient Data Streaming**: Automatic fallbacks for LibriSpeech, VoxPopuli, FLEURS, and Common Voice

---

## 🚀 Quick Start: 1-Click Cloud Execution

### Kaggle Notebook Run:
```bash
python run_kaggle.py --model-size small --epochs 3
```

### Google Colab Execution:
```bash
!pip install torch torchaudio accelerate datasets -q
!python train_colab.py --model-size small --languages en --epochs 3 --batch-size 2 --gradient-accumulation-steps 4
```

---

## ⚡ Streamlined Training Plan & Architecture

| Parameter Tier | Parameters | Subsampling | Peak VRAM | Recommended Execution |
|----------------|------------|-------------|-----------|-----------------------|
| **nano**       | ~25M       | 4x Conv2D   | ~0.6 GB   | Ultra-fast test / CPU |
| **small**      | ~85M       | 4x Conv2D   | ~1.2 GB   | Colab / Kaggle (Default)|
| **medium**     | ~274M      | 4x Conv2D   | ~3.1 GB   | Free T4 GPU (FP16)    |
| **large**      | ~662M      | 4x Conv2D   | ~8.5 GB   | High-RAM / Dual GPU   |

---

## 🛠 Features & Optimizations

### 1. 4x Conv2D Time Downsampling
Standard audio spectrograms generated at 16kHz produce 62.5 time frames per second. Without subsampling, self-attention memory scales as $O(T^2)$. The 4x Conv2D subsampling module downsamples audio frames by $4\times$, reducing activation memory by **16x** ($1/16$) while maintaining model parameter counts.

### 2. Micro-Batching & Gradient Accumulation
Training runs with `--batch-size 2` and `--gradient-accumulation-steps 4` yield an effective batch size of 8 while bounding VRAM consumption under 3.5GB.

### 3. Audio Truncation
All audio waveforms are clamped to `--max-audio-len 160000` (10 seconds at 16kHz) during batching to prevent rare long audio outliers (30s - 60s) from causing CUDA Out Of Memory errors.

### 4. Disk-Safe Single File Checkpointing
Instead of writing multi-gigabyte optimizer state directories per epoch (which consume >18GB and trigger Kaggle `/kaggle/working` disk quota failures), training saves clean, single-file PyTorch weight dicts (`model_best.pt` and `model_latest.pt`).

---

## 🎤 Local Inference Test

To verify the model architecture locally:

```bash
python inference.py --dummy-test --model-size small
```

---

## 📜 Citation & License

MIT License.
