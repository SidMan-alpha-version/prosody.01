# Prosody: Advanced Multilingual TTS/ASR with Free Colab Support

**274M parameter Conformer model** trained on Mozilla Common Voice with:
- ✅ Ancient Greek pronunciation markers
- ✅ Real CTC + duration + prosody losses
- ✅ Cloud streaming (no downloads!)
- ✅ Free Colab compatible (15GB GPU)
- ✅ 20 languages ready to train

## Quick Start: 1 Command

```bash
# In Colab:
!pip install torch torchaudio accelerate datasets -q
!git clone https://github.com/SidMan-alpha-version/prosody.01.git prosody
%cd prosody
!accelerate launch train_colab.py --use-streaming --languages all --epochs 5
```

## Train on Cloud Data (No Downloads!)

```python
# English only - 20-30 min per epoch
accelerate launch train_colab.py --use-streaming --languages en --epochs 5

# Multiple languages - mixed batches
accelerate launch train_colab.py --use-streaming \
    --languages en,es,fr,de,it,pt --epochs 3

# All 20 languages - ultimate multilingual (default)
accelerate launch train_colab.py --use-streaming \
    --languages all \
    --epochs 2
```

## Why This Model?

| Feature | Details |
|---------|---------|
| **Architecture** | Conformer (attention + convolution) |
| **Parameters** | 274M (fits in free Colab with FP16) |
| **Training** | 2-4 epochs per language in ~3 hours |
| **Data** | Mozilla Common Voice (20 languages) |
| **Streaming** | Cloud-powered, no storage needed |
| **Losses** | CTC + Duration + F0 + Energy + Mel |
| **Prosody** | Stress, tone, syllables, ancient Greek markers |

## Supported Languages

**From Mozilla Common Voice (20):**
`en, es, fr, de, it, pt, nl, sv, uk, el, ro, af, pl, ru, zh, hi, bn, ta, ml, ar`

**Plus (custom datasets):**
`grc (Ancient Greek), ur, gu, kn, mr, or, pa, te, sa, yue`

## Model Sizes

- **small** (85M) - Ultra-lightweight, Colab-friendly
- **medium** (274M) - Recommended, best quality
- **large** (662M) - Higher quality, needs more memory

## Requirements

```
torch>=2.0.0
torchaudio>=2.0.0
accelerate>=0.25.0
datasets>=2.14.0
transformers>=4.35.0
```

Install: `pip install -r requirements.txt`

## Training in Colab

### Setup
```python
# Cell 1: Mount drive & install
from google.colab import drive
drive.mount('/content/drive')

!pip install torch torchaudio accelerate datasets -q
!git clone https://github.com/SidMan-alpha-version/prosody.01.git prosody
%cd prosody
```

### Train
```python
# Cell 2: Single language test (quick)
!accelerate launch train_colab.py \
    --use-streaming --languages en \
    --max-samples 100 --epochs 1

# Cell 3: Full training (after test passes)
!accelerate launch train_colab.py \
    --use-streaming \
    --languages en,es,fr,de \
    --epochs 5 \
    --output-dir /content/drive/MyDrive/outputs
```

## Key Features

### ✨ Cloud Streaming
- Stream from Mozilla S3 directly to GPU
- No local storage needed
- Perfect for Colab unlimited bandwidth

### 🎵 Prosody Markers
- Stress: `ˈ` (primary), `ˌ` (secondary)
- Tone: `´` (acute), `` ` `` (grave), `˜` (circumflex)
- Syllables: `.` (boundary), `¯` (long), `˘` (short)
- Ancient Greek polytonic accents

### ⚡ Memory Optimized
- FP16 mixed precision training
- Gradient accumulation (4 steps)
- Gradient checkpointing enabled
- Fits in 15GB free Colab GPU

### 🔄 Multi-Language
- Train on any combination of 20 languages
- Interleaved batches for diversity
- Language-agnostic loss weights

## Output

Checkpoints saved every 100 steps:
```
outputs_colab/
├── epoch_1/
│   ├── pytorch_model.bin
│   ├── optimizer.bin
│   └── scheduler.bin
├── epoch_2/
└── metrics_summary.json
```

## Performance

Estimated on free Colab (T4 GPU, 15GB memory):

| Languages | Batch | Epoch | Time |
|-----------|-------|-------|------|
| 1 (en) | 4 | 1 | 20-30 min |
| 4 (en,es,fr,de) | 4 | 1 | 45 min |
| 20 (all) | 4 | 1 | 2-3 hours |

## Citation

If you use this model, please cite:

```bibtex
@software{prosody2026,
  title={Prosody: Multilingual TTS/ASR with Free Colab Support},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/prosody}
}
```

## License

MIT

## References

- [Mozilla Common Voice](https://commonvoice.mozilla.org/)
- [HuggingFace Accelerate](https://huggingface.co/docs/accelerate/)
- [Conformer Architecture](https://arxiv.org/abs/2005.08100)
- [CTC Loss](https://towardsdatascience.com/loss-function-connectionist-temporal-classification-ctc-for-speech-recognition-d99fb2c4c4ab)
