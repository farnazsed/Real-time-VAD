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

## 1. Offline & Online Evaluation (AMI Dataset)


- Evaluates VAD on pre-annotated audio files from the AMI corpus.
- Compares:
  - Offline mode (full waveform inference)
  - Online mode (chunked inference with hangover)
- Plots:
  - Ground truth vs. predictions
  - Frame-level speech probabilities
- Outputs metrics like DER, F1, precision, recall, and RTF.

### Setup Instructions

1. Clone the AMI evaluation protocol:

```bash
git clone https://github.com/pyannote/AMI-diarization-setup.git

```bash
python offline_online_evaluation.py
```

![Offline vs Online VAD Demo](assets/offline_online_evaluation.png)


---

## 2. Real-Time File-Based Visualization

- Processes local audio files in real time (e.g., `.wav`) in streaming chunks
- Uses `streamz` for live signal flow and `matplotlib` for dynamic visualization
- Displays:
  - Audio waveform (amplitude over time)
  - Frame-level speech probability (VAD scores)
  - Highlighted regions for detected speech
- Requires: `ipywidgets`, `streamz`, `matplotlib`, `pyannote.audio`, `torchaudio`

### 🛠 Dependencies

Install required libraries:

```bash
pip install pyannote.audio torchaudio matplotlib streamz ipywidgets

```bash
python realtime_file_visualizer.py
```

![Streaming VAD Demo](assets/realtime_file_visualizer.png)


---


## 3. Live Microphone Chunk-Level Detection

- Captures audio from the browser’s microphone (via JavaScript bridge in Google Colab)
- Performs real-time chunk-level voice activity detection (32ms resolution)
- Displays:
  - Live speech status ("SPEECH DETECTED" / "SILENCE")
  - Aggregated speech segments on waveform
  - Speech vs. silence bar chart

> ⚠️ This script is designed to be run in **Google Colab**, not locally.  
> Make sure your browser **allows microphone access** when prompted.


![Live VAD Demo](assets/mic_live_chunk_detector.png)

---

## Setup

1. Install dependencies:

```bash
pip install pyannote.audio torchaudio streamz matplotlib ipywidgets
apt-get install libportaudio2
```

2. Authenticate with Hugging Face:

```python
from huggingface_hub import notebook_login
notebook_login()
```



---

##  Hugging Face VAD Model

All scripts use the pretrained model:  
[`pyannote/voice-activity-detection`](https://huggingface.co/pyannote/voice-activity-detection)

---

## Example Output

- DER, F1, and frame-level accuracy plots (for AMI files)
- Real-time waveform + VAD probability visualization
- Microphone-driven segment detection with speech/silence durations

---

