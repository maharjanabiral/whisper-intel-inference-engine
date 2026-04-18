import numpy as np
import librosa
import time

SAMPLE_RATE    = 16000
CHUNK_SECONDS  = 30
OVERLAP_SECONDS = 5
CHUNK_SAMPLES  = SAMPLE_RATE * CHUNK_SECONDS
OVERLAP_SAMPLES = SAMPLE_RATE * OVERLAP_SECONDS

def load_audio(path: str):
    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return audio

def split_audio(audio):
    """Split audio into overlapping 30 second chunks"""
    chunks = []
    step = CHUNK_SAMPLES - OVERLAP_SAMPLES
    start = 0

    while start < len(audio):
        end = start + CHUNK_SAMPLES
        chunk = audio[start:end]

        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
        chunks.append(chunk)
        start += step

    return chunks

def remove_overlap(texts: list[str]) -> str:
    """
    Merge chunk transcriptions by removing duplicated overlap text.
    Strategy: find the longest common suffix/prefix between consecutive chunks.
    """
    if not texts:
        return ""
    if len(texts) == 1:
        return texts[0]

    merged = texts[0]

    for next_text in texts[1:]:
        if not next_text.strip():
            continue

        words_merged = merged.split()
        words_next   = next_text.split()

        best_overlap = 0
        max_check = min(len(words_merged), len(words_next), 30)

        for n in range(max_check, 0, -1):
            suffix = " ".join(words_merged[-n:]).lower().strip(".,!?")
            prefix = " ".join(words_next[:n]).lower().strip(".,!?")
            if suffix == prefix:
                best_overlap = n
                break

        if best_overlap > 0:
            remainder = " ".join(words_next[best_overlap:])
            if remainder.strip():
                merged = merged + " " + remainder
        else:
            merged = merged + " " + next_text

    return merged.strip()


def transcribe_file(
    audio_path: str,
    engine,
    preprocessor,
) -> dict:
    """Transcribe any length audio file."""
    audio = load_audio(audio_path)
    chunks = split_audio(audio)
    t0 = time.time()
    chunk_texts = []

    for i, chunk in enumerate(chunks):
        mel    = preprocessor.audio_to_mel(chunk)
        result = engine.transcribe(mel)
        chunk_texts.append(result['text'])
    final_text = remove_overlap(chunk_texts)
    total_time = round((time.time() - t0) * 1000)

    return {
        "text":        final_text,
    }

