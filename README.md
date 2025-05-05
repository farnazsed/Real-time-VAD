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
