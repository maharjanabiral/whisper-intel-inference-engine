import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from transformers import WhisperProcessor

SAMPLE_RATE = 16000
CHUNK_LENGTH_S = 30
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_LENGTH_S
N_MEL_BINS = 80
N_FRAMES = 3000

class AudioPreprocessor:
    def __init__(self, model_dir: str | Path):
        self.model_dir = model_dir
        self.processor = WhisperProcessor.from_pretrained(self.model_dir)
        self.feature_extractor = self.processor.feature_extractor

    def load_audio(self, audio_path: str | Path):
        """Load any audio file and convert to 16kHz mono float32"""
        audio_path = Path(audio_path)
        audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
        return audio
    
    def audio_to_mel(self, audio):
        """Converts raw audio array to log-mel spectrogram tensor"""
        if len(audio) < CHUNK_SAMPLES:
            audio = np.pad(audio, (0, CHUNK_SAMPLES - len(audio)))
        else:
            audio = audio[:CHUNK_SAMPLES]

        inputs = self.feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="np"
        )
        mel = inputs.input_features
        return mel
    
    def process_file(self, audio_path: str | Path):
        audio = self.load_audio(audio_path)
        mel = self.audio_to_mel(audio)
        return mel
    