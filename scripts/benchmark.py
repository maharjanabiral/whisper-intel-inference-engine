import time
import numpy as np
from ov_whisper.transcribe import transcribe_file
from ov_whisper.engine import WhisperEngine
from ov_whisper.audio import AudioPreprocessor
from faster_whisper import WhisperModel
import matplotlib.pyplot as plt

AUDIO_FILE = "audio/test_short.wav"
MODEL_SIZE = "small"
MODEL_DIR  = "huggingface_models/whisper-small-ov"
DEVICE = "GPU"

print("=" * 50)
print("OV-Whisper (OpenVINO + Intel Arc GPU)")
print("=" * 50)


engine = WhisperEngine(MODEL_DIR, DEVICE)
preprocessor = AudioPreprocessor(MODEL_DIR)


times = []
for i in range(5):
    t0 = time.time()
    result = transcribe_file(AUDIO_FILE, engine, preprocessor)
    times.append(time.time() - t0)

ov_avg = np.mean(times)
ov_text = result["text"]
print(f"  Text    : {ov_text}")
print(f"  Avg     : {ov_avg*1000:.0f}ms over 5 runs")
print(f"  RTF     : {ov_avg/25:.2f}x  (lower is better)")

print("=" * 50)
print("Faster-Whisper (CTranslate2 + CPU)")
print("=" * 50)


fw_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

segments, _ = fw_model.transcribe(AUDIO_FILE)
list(segments)

times = []
for i in range(5):
    t0 = time.time()
    segments, _ = fw_model.transcribe(AUDIO_FILE, beam_size=1)
    fw_text = " ".join(s.text for s in segments)
    times.append(time.time() - t0)

fw_avg = np.mean(times)
print(f"  Text    : {fw_text}")
print(f"  Avg     : {fw_avg*1000:.0f}ms over 5 runs")
print(f"  RTF     : {fw_avg/25:.2f}x  (lower is better)")

print("=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"  OV-Whisper  : {ov_avg*1000:.0f}ms  (RTF {ov_avg/25:.2f}x)")
print(f"  FasterWhisp : {fw_avg*1000:.0f}ms  (RTF {fw_avg/25:.2f}x)")
print(f"  Speedup     : {fw_avg/ov_avg:.1f}x faster than faster-whisper")
print("=" * 50)

import matplotlib.pyplot as plt

PURPLE = "#7F77DD"
GRAY   = "#888780"

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(
    ["OV-Whisper\n(Arc GPU)", "Faster-Whisper\n(CPU)"],
    [ov_avg * 1000, fw_avg * 1000],
    color=[PURPLE, GRAY],
    width=0.4
)
for bar, val in zip(bars, [ov_avg * 1000, fw_avg * 1000]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
            f"{val:.0f}ms", ha="center", fontweight="bold")

ax.set_title(f"OV-Whisper vs Faster-Whisper  —  {fw_avg/ov_avg:.1f}x speedup")
ax.set_ylabel("Average latency (ms)")
ax.set_ylim(0, max(ov_avg, fw_avg) * 1000 * 1.3)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("scripts/benchmark_results.png", dpi=150)
plt.show()