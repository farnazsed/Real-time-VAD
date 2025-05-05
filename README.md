#  Real-Time & Offline Voice Activity Detection (VAD) using pyannote.audio

This repository contains three Python-based implementations for performing Voice Activity Detection (VAD) using [pyannote.audio](https://github.com/pyannote/pyannote-audio). Each implementation targets a different use case, from evaluating performance on the AMI dataset to real-time microphone input with live visualization.


---

## Contents

| Script | Description |
|--------|-------------|
| `1_offline_online_evaluation.py` | Offline & online evaluation on AMI dataset with detection metrics (DER, F1, precision, etc.) |
| `2_realtime_file_visualizer.py` | Real-time VAD processing from audio files using Streamz & matplotlib |
| `3_mic_live_chunk_detector.py` | Live microphone-based VAD with JavaScript interface and chunk-level decisions |

---

## TL;DR

1. Install [`pyannote.audio`](https://github.com/pyannote/pyannote-audio) with `pip install pyannote.audio`
2. Accept [`pyannote/segmentation-3.0`](https://hf.co/pyannote/segmentation-3.0) user conditions
3. Create access token at [`hf.co/settings/tokens`](https://hf.co/settings/tokens).

```python
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/segmentation-3.0",
    use_auth_token="HUGGINGFACE_ACCESS_TOKEN_GOES_HERE")

# send pipeline to GPU (when available)
import torch
pipeline.to(torch.device("cuda"))
```

## 🧪 1. Offline & Online Evaluation (AMI Dataset)

- Evaluates VAD on pre-annotated audio files from the AMI corpus.
- Compares:
  - Offline mode (full waveform inference)
  - Online mode (chunked inference with hangover)
- Plots:
  - Ground truth vs. predictions
  - Frame-level speech probabilities
- Outputs metrics like DER, F1, precision, recall, and MACs/frame.

```bash
python 1_offline_online_evaluation.py
```
