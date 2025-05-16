# Without online plots
!pip install pyannote.audio torchaudio matplotlib soundfile
!apt-get install libportaudio2

from huggingface_hub import notebook_login
notebook_login()

import torch
import numpy as np
from pyannote.audio import Pipeline
from google.colab import output
from IPython.display import display, Javascript, HTML, clear_output, Audio
import time
from ipywidgets import widgets
import queue
import threading
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import soundfile as sf
import io
from matplotlib.animation import FuncAnimation

class ContinuousSpeechDetector:
    def __init__(self):
        self.sample_rate = 16000
        self.chunk_size = 512  # ~128ms chunks
        self.vad_window = 2048  # Window size for VAD processing
        self.hangover_time = 1
        self.last_speech_time = 0
        self.plot_refresh_rate = 0.5  # seconds between plot updates
        self.last_plot_update = 0

        self.pipeline = Pipeline.from_pretrained(
            "pyannote/voice-activity-detection",
            use_auth_token=True
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self.pipeline.instantiate({
            'onset': 0.7, 'offset': 0.5,
            'min_duration_on': 0.1,
            'min_duration_off': 0.2
        })

        self.is_speaking = False
        self.is_recording = False
        self.audio_buffer = np.zeros(self.vad_window, dtype=np.float32)
        self.full_audio = np.array([], dtype=np.float32)
        self.vad_results = []
        self.audio_queue = queue.Queue()
        self.status_display = widgets.Output()
        self.plot_display = widgets.Output()
        self.audio_display = widgets.Output()

        # For real-time plotting
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.line, = self.ax.plot([], [], 'b-', alpha=0.7, linewidth=0.5)
        self.ax.set_xlim(0, 5)  # Start with 5-second window
        self.ax.set_ylim(-1, 1)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title("Real-time Speech Detection")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.poly_collection = None
        plt.close(self.fig)  # We'll display it in the widget

        self.setup_ui()

        self.processing_thread = None

    def setup_ui(self):
        """Initialize user interface"""
        self.start_button = widgets.Button(
            description="Start Detection",
            button_style='success',
            layout=widgets.Layout(width='150px')
        )
        self.stop_button = widgets.Button(
            description="Stop Detection",
            button_style='danger',
            layout=widgets.Layout(width='150px')
        )
        self.plot_button = widgets.Button(
            description="Show Results",
            button_style='info',
            layout=widgets.Layout(width='150px'),
            disabled=True
        )
        self.save_button = widgets.Button(
            description="Save Audio",
            button_style='warning',
            layout=widgets.Layout(width='150px'),
            disabled=True
        )
        self.play_button = widgets.Button(
            description="Play Audio",
            button_style='primary',
            layout=widgets.Layout(width='150px'),
            disabled=True
        )

        self.start_button.on_click(self.start_detection)
        self.stop_button.on_click(self.stop_detection)
        self.plot_button.on_click(self.show_results)
        self.save_button.on_click(self.save_audio)
        self.play_button.on_click(self.play_audio)

        display(widgets.HBox([
            self.start_button,
            self.stop_button,
            self.plot_button,
            self.save_button,
            self.play_button
        ]))
        display(self.status_display)
        display(self.plot_display)
        display(self.audio_display)

        with self.status_display:
            display(HTML("""
            <div style="font-size:24px; color:blue; text-align:center;
                        padding:20px; background-color:#e6f3ff; border-radius:10px;">
                Ready to start detection
            </div>
            """))

    def process_audio_chunks(self):
        while self.is_recording:
            try:
                audio_data = self.audio_queue.get(timeout=0.1)

                audio_array = np.array(audio_data, dtype=np.float32)
                self.audio_buffer[:-len(audio_array)] = self.audio_buffer[len(audio_array):]
                self.audio_buffer[-len(audio_array):] = audio_array

                self.full_audio = np.concatenate((self.full_audio, audio_array))

                self._run_vad()

                # Update plot periodically
                current_time = time.time()
                if current_time - self.last_plot_update > self.plot_refresh_rate:
                    self.last_plot_update = current_time
                    self._update_plot()

            except queue.Empty:
                continue

    def _run_vad(self):
        current_rms = np.sqrt(np.mean(self.audio_buffer**2))

        # Only process if above noise floor
        if current_rms < 0.01:  # Typical silent RMS threshold
            return
        try:
            with torch.no_grad():
                vad_out = self.pipeline({
                    "waveform": torch.from_numpy(self.audio_buffer).float().unsqueeze(0),
                    "sample_rate": self.sample_rate
                })

            current_time = len(self.full_audio) / self.sample_rate
            window_duration = self.vad_window / self.sample_rate

            speech_now = False
            for segment in vad_out.get_timeline().support():
                start = current_time - window_duration + segment.start
                end = current_time - window_duration + segment.end
                self.vad_results.append((start, end))
                speech_now = True

            if speech_now:
                self.last_speech_time = current_time
            elif (current_time - self.last_speech_time) < self.hangover_time:
                self.vad_results.append((self.last_speech_time, current_time))
                speech_now = True

            if speech_now != self.is_speaking:
                self.is_speaking = speech_now
                self._update_display()

        except Exception as e:
            print(f"VAD error: {str(e)}")

    def _update_display(self):
        """Update the status display"""
        with self.status_display:
            clear_output(wait=True)
            if self.is_speaking:
                display(HTML("""
                <div style="font-size:48px; color:green; text-align:center;
                            padding:20px; background-color:#e6ffe6; border-radius:10px;">
                    SPEECH DETECTED
                </div>
                <div style="color:gray; text-align:center;">
                    Last update: {time}
                </div>
                """.format(time=time.strftime('%H:%M:%S'))))
            else:
                display(HTML("""
                <div style="font-size:48px; color:red; text-align:center;
                            padding:20px; background-color:#ffe6e6; border-radius:10px;">
                    SILENCE
                </div>
                <div style="color:gray; text-align:center;">
                    Last update: {time}
                </div>
                """.format(time=time.strftime('%H:%M:%S'))))

    def _update_plot(self):
        """Update the real-time plot"""
        if len(self.full_audio) == 0:
            return

        duration = len(self.full_audio) / self.sample_rate
        time_axis = np.linspace(0, duration, len(self.full_audio))

        # Merge overlapping/adjacent segments (including hangover periods)
        merged_segments = []
        for start, end in sorted(self.vad_results):
            if not merged_segments:
                merged_segments.append([start, end])
            else:
                last_start, last_end = merged_segments[-1]
                if start <= (last_end + self.hangover_time):  # Apply hangover threshold
                    merged_segments[-1][1] = max(last_end, end)  # Extend segment
                else:
                    merged_segments.append([start, end])

        # Update plot
        self.line.set_data(time_axis, self.full_audio)

        # Adjust x-axis to show most recent 5 seconds (or full duration if less)
        view_window = min(5, duration)
        self.ax.set_xlim(max(0, duration - view_window), duration)

        # Update speech segments visualization
        if self.poly_collection:
            self.poly_collection.remove()

        if merged_segments:
            verts = [
                [(start, -1), (start, 1), (end, 1), (end, -1)]
                for start, end in merged_segments
                if end > (duration - view_window)  # Only show segments in current view
            ]
            if verts:
                self.poly_collection = PolyCollection(
                    verts, facecolors='green', alpha=0.3, edgecolors='none'
                )
                self.ax.add_collection(self.poly_collection)

        with self.plot_display:
            clear_output(wait=True)
            display(self.fig)

    def audio_callback(self, audio_data):
        """Callback for incoming audio data"""
        if self.is_recording:
            self.audio_queue.put(audio_data)

    def start_detection(self, button=None):
        """Start voice activity detection"""
        if not self.is_recording:
            self.is_recording = True
            self.is_speaking = False
            self.audio_buffer.fill(0)
            self.full_audio = np.array([], dtype=np.float32)
            self.vad_results = []
            self.plot_button.disabled = True
            self.save_button.disabled = True
            self.play_button.disabled = True

            self.processing_thread = threading.Thread(target=self.process_audio_chunks)
            self.processing_thread.start()

            self._update_display()
            self._start_microphone()

    def stop_detection(self, button=None):
        """Stop voice activity detection"""
        if self.is_recording:
            self.is_recording = False
            if self.processing_thread:
                self.processing_thread.join()

            with self.status_display:
                clear_output(wait=True)
                display(HTML("""
                <div style="font-size:24px; color:blue; text-align:center;
                            padding:20px; background-color:#e6f3ff; border-radius:10px;">
                    Detection Stopped
                </div>
                """))

            self.plot_button.disabled = False
            self.save_button.disabled = False
            self.play_button.disabled = False

    def show_results(self, button=None):
        """Display the waveform and VAD results with hangover processing"""
        if len(self.full_audio) == 0:
            with self.plot_display:
                clear_output(wait=True)
                print("No audio data to display")
            return

        with self.plot_display:
            clear_output(wait=True)

            duration = len(self.full_audio) / self.sample_rate
            time_axis = np.linspace(0, duration, len(self.full_audio))

            # 1. Merge overlapping/adjacent segments (including hangover periods)
            merged_segments = []
            for start, end in sorted(self.vad_results):
                if not merged_segments:
                    merged_segments.append([start, end])
                else:
                    last_start, last_end = merged_segments[-1]
                    if start <= (last_end + self.hangover_time):  # Apply hangover threshold
                        merged_segments[-1][1] = max(last_end, end)  # Extend segment
                    else:
                        merged_segments.append([start, end])

            # 2. Filter out very short segments
            min_segment_length = 0.1
            filtered_segments = [
                (start, end) for start, end in merged_segments
                if (end - start) >= min_segment_length
            ]

            # 3. Calculate speech statistics
            speech_duration = sum(end - start for start, end in filtered_segments)
            speech_percentage = (speech_duration / duration) * 100
            segment_count = len(filtered_segments)

            fig, (ax_wave, ax_stats) = plt.subplots(
                2, 1,
                figsize=(14, 8),
                gridspec_kw={'height_ratios': [3, 1]}
            )

            ax_wave.plot(time_axis, self.full_audio, 'b-', alpha=0.7, linewidth=0.5)

            if filtered_segments:
                verts = [
                    [(start, -1), (start, 1), (end, 1), (end, -1)]
                    for start, end in filtered_segments
                ]
                ax_wave.add_collection(
                    PolyCollection(verts, facecolors='green', alpha=0.3, edgecolors='none')
                )

            ax_wave.set_title(
                f"Waveform with VAD Results\n"
                f"Speech: {speech_percentage:.1f}% | Segments: {segment_count} | Hangover: {self.hangover_time*1000:.0f}ms"
            )
            ax_wave.set_ylabel("Amplitude")
            ax_wave.set_xlim(0, duration)
            ax_wave.grid(True, alpha=0.3)

            ax_stats.barh(
                ['Speech', 'Silence'],
                [speech_duration, duration - speech_duration],
                color=['green', 'red']
            )
            ax_stats.set_xlabel("Duration (seconds)")
            ax_stats.set_title("Speech/Silence Distribution")

            for i, (label, val) in enumerate(zip(
                ['Speech', 'Silence'],
                [speech_duration, duration - speech_duration]
            )):
                ax_stats.text(
                    val/2, i,
                    f"{val:.2f}s ({val/duration:.1%})",
                    ha='center', va='center',
                    color='white', weight='bold'
                )

            plt.tight_layout()
            plt.show()

            print("\n=== VAD PERFORMANCE ===")
            print(f"Total duration: {duration:.2f} seconds")
            print(f"Speech detected: {speech_duration:.2f}s ({speech_percentage:.1f}%)")
            print(f"Number of segments: {segment_count}")
            if segment_count > 0:
                avg_length = speech_duration / segment_count
                print(f"Average segment length: {avg_length:.2f}s")
            print(f"Hangover duration: {self.hangover_time:.3f}s")

    def save_audio(self, button=None):
        """Save the recorded audio as a WAV file"""
        if len(self.full_audio) == 0:
            with self.audio_display:
                clear_output(wait=True)
                print("No audio to save")
            return

        with self.audio_display:
            clear_output(wait=True)
            print("Saving audio...")

            # Create a bytes buffer to hold the WAV data
            buffer = io.BytesIO()
            sf.write(buffer, self.full_audio, self.sample_rate, format='WAV')
            buffer.seek(0)

            # Create download link
            from IPython.display import FileLink
            display(FileLink(buffer, filename="recorded_audio.wav", result_html_prefix="Download: "))
            print("Audio saved as recorded_audio.wav")

    def play_audio(self, button=None):
        """Play the recorded audio"""
        if len(self.full_audio) == 0:
            with self.audio_display:
                clear_output(wait=True)
                print("No audio to play")
            return

        with self.audio_display:
            clear_output(wait=True)
            display(Audio(self.full_audio, rate=self.sample_rate, autoplay=True))
            print("Playing recorded audio...")

    def _start_microphone(self):
        """Initialize microphone capture with JavaScript"""
        display(Javascript("""
        let audioContext, processor;

        async function startVAD() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        sampleRate: 16000,
                        noiseSuppression: true,
                        echoCancellation: true
                    }
                });

                audioContext = new AudioContext({ sampleRate: 16000 });
                const source = audioContext.createMediaStreamSource(stream);
                processor = audioContext.createScriptProcessor(2048, 1, 1);

                processor.onaudioprocess = (e) => {
                    const data = e.inputBuffer.getChannelData(0);
                    google.colab.kernel.invokeFunction(
                        'process_audio',
                        [Array.from(data)],
                        {}
                    );
                };

                source.connect(processor);
                processor.connect(audioContext.destination);

            } catch (error) {
                console.error("Microphone error:", error);
                alert("Microphone access denied. Please refresh and click Allow when prompted");
            }
        }

        window.keepAlive = setInterval(() => {
            console.log("Keeping microphone alive");
        }, 5000);

        startVAD();
        """))


def main():
    detector = ContinuousSpeechDetector()
    output.enable_custom_widget_manager()
    output.register_callback('process_audio', detector.audio_callback)

    display(HTML("""
    <div style="margin:20px 0; padding:10px; background:#f0f0f0; border-radius:5px;">
        <h3>Continuous Speech Detection</h3>
        <p>This version provides real-time visualization and audio saving:</p>
        <ol>
            <li>Click "Start Detection"</li>
            <li>Allow microphone access</li>
            <li>See real-time detection in the plot</li>
            <li>Click "Stop Detection" when done</li>
            <li>Click "Show Results" for detailed analysis</li>
            <li>Click "Save Audio" to download recording</li>
            <li>Click "Play Audio" to hear the recording</li>
        </ol>
    </div>
    """))

if __name__ == "__main__":
    main()
