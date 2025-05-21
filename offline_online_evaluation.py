# Install required packages
!pip install pyannote.audio torchaudio matplotlib tqdm
!apt-get -qq install libportaudio2

# Authenticate with Hugging Face
from huggingface_hub import notebook_login
notebook_login()

# Now import everything else
import torch
import numpy as np
from pyannote.audio import Pipeline, Audio, Model
from pyannote.database import registry, FileFinder
from pyannote.core import Segment, Annotation, notebook, SlidingWindow, SlidingWindowFeature
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.metrics.detection import DetectionErrorRate
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import time
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Clone AMI dataset setup
!git clone https://github.com/pyannote/AMI-diarization-setup.git

# Set up the database
registry.load_database("AMI-diarization-setup/pyannote/database.yml")
preprocessors = {"audio": FileFinder()}
ami = registry.get_protocol('AMI.SpeakerDiarization.mini', preprocessors=preprocessors)



class VADEvaluator:
    def __init__(self):
        self.sample_rate = 16000
        self.hop_size = 16384
        self.vad_window = 65536
        self.hangover_time = 0.4

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize pipeline with authentication
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/voice-activity-detection@2.1",
            use_auth_token=True
        ).to(self.device)

        self.pipeline.instantiate({
            'onset': 0.7, 'offset': 0.3,
            'min_duration_on': 0.1,
            'min_duration_off': 0.2
        })

        self.audio = Audio(sample_rate=self.sample_rate, mono=True)
        self.der_metric = DiarizationErrorRate()
        self.detection_metric = DetectionErrorRate()
        self.processing_times = []
        self.frame_counts = []

    def evaluate_offline(self, test_file):
        """Evaluate VAD performance on a complete audio file"""
        # Load audio and ensure proper shape (channel, time)
        waveform, _ = self.audio(test_file)
        waveform = waveform.unsqueeze(0) if len(waveform.shape) == 1 else waveform  # Ensure (channel, time)
        reference = test_file['annotation']

        # Process entire file at once
        start_time = time.time()
        vad_result = self.pipeline({
            "waveform": waveform.to(self.device),
            "sample_rate": self.sample_rate
        })
        processing_time = time.time() - start_time

        # Calculate metrics
        der = self.der_metric(reference, vad_result, uem=test_file['annotated'])
        detection_rate = self.detection_metric(reference, vad_result)

        return {
            'der': der,
            'detection_rate': detection_rate,
            'processing_time': processing_time,
            'audio_duration': waveform.shape[1]/self.sample_rate,
            'rtf': processing_time / (waveform.shape[1]/self.sample_rate),
            'waveform': waveform,
            'vad_result': vad_result,
            'reference': reference
        }

    def evaluate_online(self, test_file):
        """Simulate online processing by feeding audio in chunks"""
        waveform, _ = self.audio(test_file)
        waveform = waveform.unsqueeze(0) if len(waveform.shape) == 1 else waveform  # Ensure (channel, time)
        reference = test_file['annotation']

        # Reset timing stats
        self.processing_times = []
        self.frame_counts = []

        # Simulate online processing
        start_time = time.time()
        vad_result, frame_scores, total_macs, macs_per_second  = self._process_online(waveform)
        processing_time = time.time() - start_time

        # Calculate metrics
        der = self.der_metric(reference, vad_result, uem=test_file['annotated'])
        detection_rate = self.detection_metric(reference, vad_result)

        

        return {
            'der': der,
            'detection_rate': detection_rate,
            'processing_time': processing_time,
            'audio_duration': waveform.shape[1]/self.sample_rate,
            'rtf': processing_time / (waveform.shape[1]/self.sample_rate),
            'avg_frame_time': np.mean(self.processing_times),
            'waveform': waveform,
            'vad_result': vad_result,
            'vad_frame_scores': frame_scores,
            'reference': reference,
            'total_macs': total_macs,
            'macs_per_second': macs_per_second
        }

    def _process_online(self, waveform):
        waveform = waveform.to(self.device)
        results = []
        frame_scores = []
        frame_times = []

        num_channels, num_samples = waveform.shape
        num_chunks = max(1, (num_samples - self.vad_window) // self.hop_size + 1)

        
        # Total MACs = MACs per window × number of windows
        total_macs = num_chunks * 438000
        
        # MACs per second (throughput)
        duration_seconds = num_samples / 16000
        macs_per_second = total_macs / duration_seconds
            
        

        # Create proper sliding window resolution
        resolution = SlidingWindow(
            start=0,
            duration=self.vad_window / self.sample_rate,
            step=self.hop_size / self.sample_rate
        )

        # State tracking
        speech_buffer = 0
        current_segment = None
        last_speech_time = 0

        for i in range(num_chunks):
            current_time = i * self.hop_size / self.sample_rate
            start = i * self.hop_size
            end = start + self.vad_window

            # Handle last chunk
            chunk = waveform[:, start:end]
            if chunk.shape[1] < self.vad_window:
                padding = self.vad_window - chunk.shape[1]
                chunk = torch.nn.functional.pad(chunk, (0, padding))

            # VAD Prediction
            with torch.no_grad():
                vad_out = self.pipeline({
                    "waveform": chunk,
                    "sample_rate": self.sample_rate
                })

            # Frame time is now properly aligned with resolution
            frame_time = resolution[i].middle
            frame_times.append(frame_time)

            # Hysteresis smoothing (5/7 frame decision)
            is_speech = bool(list(vad_out.get_timeline().support()))
            if is_speech:
                speech_buffer = min(speech_buffer + 1, 7)
                last_speech_time = frame_time
            else:
                speech_buffer = max(speech_buffer - 1, 0)

            frame_score = 1.0 if speech_buffer >= 5 else 0.0
            frame_scores.append([frame_score])

            # Segment management with proper time alignment
            if frame_score == 1.0:
                if current_segment is None:
                    segment_start = resolution[i].start  # Use window start time
                    current_segment = [segment_start, frame_time, "speech"]
                else:
                    current_segment[1] = frame_time
            else:
                if current_segment is not None:
                    if frame_time <= last_speech_time + self.hangover_time:
                        current_segment[1] = frame_time
                    else:
                        if (current_segment[1] - current_segment[0]) >= 0.1:
                            results.append(tuple(current_segment))
                        current_segment = None

        # Finalize last segment if valid
        if current_segment is not None and (current_segment[1] - current_segment[0]) >= 0.1:
            results.append(tuple(current_segment))

        # Merge segments with proper time handling
        merged_results = []
        for seg in sorted(results, key=lambda x: x[0]):
            if merged_results and seg[0] <= merged_results[-1][1] + 0.2:
                merged_results[-1] = (merged_results[-1][0], max(merged_results[-1][1], seg[1]), "speech")
            else:
                merged_results.append(seg)

        # Create output annotation
        annotation = Annotation()
        for start, end, _ in merged_results:
            annotation[Segment(start, end)] = "speech"

        # Create frame-level output with proper resolution
        frame_level_result = SlidingWindowFeature(
            np.array(frame_scores),
            SlidingWindow(
                start=0,
                duration=resolution.step,  # Frame duration matches hop size
                step=resolution.step       # Step size same as duration for non-overlapping frames
            )
        )
        

        return annotation, frame_level_result, total_macs, macs_per_second

    def plot_vad_comparison(self, test_file, results):
        """Plot VAD results vs ground truth and frame-level prediction"""

        # Get ground truth and VAD results
        ground_truth = test_file["annotation"]
        vad_annotation = results['vad_result']  # already an Annotation
        waveform = results['waveform']
        precision, recall, f1_score, true_positives, false_positives, false_negatives = self.calculate_detection_metrics(
            ground_truth.get_timeline().support(),
            vad_annotation.get_timeline().support(),
            results['audio_duration']
        )

        # Create figure with 3 subplots
        """fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        # Plot waveform
        waveform_time = np.linspace(0, len(waveform[0]) / self.sample_rate, len(waveform[0]))
        ax1.plot(waveform_time, waveform[0].cpu().numpy(), 'b-', alpha=0.7, linewidth=0.5)
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Audio Waveform')
        ax1.grid(True, alpha=0.3)

        # Plot ground truth vs VAD result (annotation level)
        from pyannote.core import notebook
        notebook.plot_annotation(ground_truth, ax=ax2, time=True, legend=False)
        notebook.plot_annotation(vad_annotation, ax=ax2, time=True, legend=False)
        ax2.set_title('Ground Truth (Blue) vs VAD Result (Red)')
        ax2.set_xlabel('Time (seconds)')
        ax2.grid(True, alpha=0.3)

        # Plot frame-level VAD prediction (from SlidingWindowFeature)
        scores = results['vad_frame_scores']
        times = [t for t in scores.sliding_window]
        values = scores.data.squeeze()

        ax3.plot(times, values, label="Online Frame-Level VAD", color='purple')
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Speech Probability")
        ax3.set_title("Frame-wise VAD Prediction (Online Mode)")
        ax3.set_ylim(-0.1, 1.1)
        ax3.grid(True)
        ax3.legend()

        plt.tight_layout()
        plt.show()"""

        return precision, recall, f1_score, true_positives, false_positives, false_negatives

      # Calculate precision, recall, F1


    def calculate_detection_metrics(self, ground_truth, vad_result, audio_duration):
        """Calculate precision, recall and F1-score from segments"""
        # Convert to binary arrays at 100ms resolution
        time_resolution = 0.1  # 100ms
        num_points = int(audio_duration / time_resolution)
        time_points = np.linspace(0, audio_duration, num_points)

        # Create binary arrays
        gt_binary = np.zeros(num_points)
        vad_binary = np.zeros(num_points)

        # Mark ground truth segments
        for segment in ground_truth:
            start_idx = int(segment.start / time_resolution)
            end_idx = int(segment.end / time_resolution)
            gt_binary[start_idx:end_idx] = 1

        # Mark detected segments
        for segment in vad_result:
            start_idx = int(segment.start / time_resolution)
            end_idx = int(segment.end / time_resolution)
            vad_binary[start_idx:end_idx] = 1

        # Calculate metrics
        true_positives = np.sum((gt_binary == 1) & (vad_binary == 1))
        false_positives = np.sum((gt_binary == 0) & (vad_binary == 1))
        false_negatives = np.sum((gt_binary == 1) & (vad_binary == 0))

        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)



        return precision, recall, f1_score, true_positives, false_positives, false_negatives

def run_full_evaluation():
    evaluator = VADEvaluator()
    test_files = list(ami.test())[:3]  # Evaluate on first 5 files

    offline_results = []
    online_results = []
    precisions = []
    recalls = []
    f1_scores  = []
    true_positives_l = []
    false_positives_l = []
    false_negatives_l = []
    

    print("Running offline evaluation...")
    for test_file in tqdm(test_files, desc="Processing files"):
        result = evaluator.evaluate_offline(test_file)
        offline_results.append(result)

    print("\nRunning online evaluation...")
    for test_file in tqdm(test_files, desc="Processing files"):
        result = evaluator.evaluate_online(test_file)
        precision, recall, f1_score, true_positives, false_positives, false_negatives = evaluator.plot_vad_comparison(test_file, result)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
        true_positives_l.append(true_positives)
        false_positives_l.append(false_positives)
        false_negatives_l.append(false_negatives)
        total_macs = result["total_macs"]
        macs_per_second = result["macs_per_second"]
        online_results.append(result)



    # Print summary statistics

    
        
    def print_stats(results, mode):
        ders = [r['der'] for r in results]
        detection_rates = [r['detection_rate'] for r in results]
        rtfs = [r['rtf'] for r in results]

        print(f"\n=== {mode.upper()} MODE SUMMARY ===")
        print(f"Average DER: {np.mean(ders):.3f} ± {np.std(ders):.3f}")
        print(f"Average Detection Rate: {np.mean(detection_rates):.3f} ± {np.std(detection_rates):.3f}")
        print(f"Average RTF: {np.mean(rtfs):.3f} ± {np.std(rtfs):.3f}")

        if mode == 'online':
            frame_times = [r['avg_frame_time'] for r in results]
            print(f"Average Frame Processing Time: {np.mean(frame_times):.5f}s ± {np.std(frame_times):.5f}s")


    print_stats(offline_results, 'offline')
    print_stats(online_results, 'online')

    # Plot results for first file
    print("\nPlotting results for first test file...")
    #evaluator.plot_vad_comparison(test_files[0], online_results[0])
    print("\nDetection Metrics:")
    print(f"Precision: {np.mean(precisions):.3f}")
    print(f"Recall: {np.mean(recalls):.3f}")
    print(f"F1-score: {np.mean(f1_scores):.3f}")
    print(f"True Positives: {np.mean(true_positives_l)}")
    print(f"False Positives: {np.mean(false_positives_l)}")
    print(f"False Negatives: {np.mean(false_negatives_l)}")
    print(f"total_macs: {total_macs}")
    print(f"macs_per_second:  {macs_per_second}")

    
    return online_results, offline_results, test_files

if __name__ == "__main__":
    online_results, offline_results, test_files =  run_full_evaluation()
