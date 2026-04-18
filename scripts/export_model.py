from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from transformers import WhisperProcessor
import openvino as ov
from pathlib import Path

model_id = "openai/whisper-small"
output_dir = Path("models/whisper-small-ov")

output_dir.mkdir(parents=True, exist_ok=True)
model = OVModelForSpeechSeq2Seq.from_pretrained(
    model_id=model_id,
    export=True,
    compile=False,
    load_in_8bit=False
)

model.save_pretrained(output_dir)
processor = WhisperProcessor.from_pretrained(model_i
processor.save_pretrained(output_dir)

print("Export complete")