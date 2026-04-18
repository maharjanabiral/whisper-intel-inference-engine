import numpy as np
import pyaudio
import webrtcvad
import collections
import time
from pathlib import Path
from ov_whisper.audio import AudioPreprocessor
from ov_whisper.engine import WhisperEngine

SAMPLE_RATE    = 16000
FRAME_DURATION = 30
FRAME_SAMPLES  = int(SAMPLE_RATE * FRAME_DURATION / 1000)
FRAME_BYTES    = FRAME_SAMPLES * 2

VAD_AGGRESSIVENESS = 2
PADDING_DURATION   = 400
PADDING_FRAMES     = PADDING_DURATION // FRAME_DURATION


MIN_SPEECH_DURATION = 0.5

class RealTimeTranscriber:
    def __init__(self, model_dir: str | Path, device: str = "GPU"):
        print("Loading engine...")
        self.preprocessor = AudioPreprocessor(model_dir)
        self.engine = WhisperEngine(model_dir, device=device)
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        print("Ready — speak into your microphone\n")

    def _transcribe_segment(self, audio_frames: list) -> str:
        """Transcribe a collected segment of speech frames."""
        # combine all frames into one array
        audio_bytes = b"".join(audio_frames)
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        audio = audio / 32768.0

        duration = len(audio) / SAMPLE_RATE
        if duration < MIN_SPEECH_DURATION:
            return ""

        mel = self.preprocessor.audio_to_mel(audio)
        result = self.engine.transcribe(mel)
        return result["text"]

    def run(self):
        """Main loop — listens to mic and prints transcriptions."""
        pa = pyaudio.PyAudio()

        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=FRAME_SAMPLES,
        )

        print("Listening... \n")

        ring_buffer = collections.deque(maxlen=PADDING_FRAMES)
        triggered = False
        voiced_frames = []

        try:
            while True:
                frame = stream.read(FRAME_SAMPLES, exception_on_overflow=False)
                is_speech = self.vad.is_speech(frame, SAMPLE_RATE)

                if not triggered:
                    ring_buffer.append((frame, is_speech))
                    num_voiced = sum(1 for _, s in ring_buffer if s)
                    if num_voiced > 0.9 * ring_buffer.maxlen:
                        triggered = True
                        print("", end="", flush=True)
                        voiced_frames = [f for f, _ in ring_buffer]
                        ring_buffer.clear()
                else:
                    voiced_frames.append(frame)
                    ring_buffer.append((frame, is_speech))
                    num_unvoiced = sum(1 for _, s in ring_buffer if not s)

                    if num_unvoiced > 0.9 * ring_buffer.maxlen:
                        triggered = False
                        t0 = time.time()
                        text = self._transcribe_segment(voiced_frames)
                        elapsed = round((time.time() - t0) * 1000)

                        if text:
                            print(f"{text}  [{elapsed}ms]")

                        voiced_frames = []
                        ring_buffer.clear()

        except KeyboardInterrupt:
            print("\n\nStopped.")
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

if __name__ == "__main__":
    MODEL_DIR = "huggingface_models/whisper-small-ov"
    DEVICE    = "GPU"

    transcriber = RealTimeTranscriber(MODEL_DIR, device=DEVICE)
    transcriber.run()