import numpy as np
import openvino as ov
import time
from pathlib import Path
from transformers import WhisperProcessor

TOKEN_SOT = 50258
TOKEN_ENGLISH = 50259
TOKEN_TRANSCRIBE = 50359
TOKEN_NO_TIMESTAMPS = 50363
TOKEN_EOT = 50257
MAX_TOKENS = 448

class WhisperEngine:
    def __init__(self, model_dir: str | Path, device: str = "GPU"):
        self.model_dir = Path(model_dir)
        self.device = device
        core = ov.Core()

        encoder_model = core.read_model(self.model_dir / "openvino_encoder_model.xml")
        self.encoder = core.compile_model(encoder_model, "CPU")
        print(f"Encoder compiled on CPU")

        decoder_model = core.read_model(self.model_dir / "openvino_decoder_model.xml")
        self.decoder = core.compile_model(decoder_model, self.device)
        print(f"Decoder compiled on {self.device}")

        self.processor = WhisperProcessor.from_pretrained(self.model_dir)
        print(f"Engine ready on {self.device}")
    
    def encode(self, mel):
        """Run Encoder"""
        result = self.encoder({"input_features": mel})
        return result["last_hidden_state"]
    
    def decode(self, audio_features):
            token_ids = [TOKEN_SOT, TOKEN_ENGLISH, TOKEN_TRANSCRIBE, TOKEN_NO_TIMESTAMPS]
            generated_tokens = []
            infer_request = self.decoder.create_infer_request()

            # Phase 1: feed all prompt tokens at once to initialize KV cache
            inputs = {
                "input_ids":             np.array([token_ids], dtype=np.int64),
                "encoder_hidden_states": audio_features,
                "beam_idx":              np.array([0], dtype=np.int32),
            }
            infer_request.infer(inputs)
            logits = infer_request.get_output_tensor(0).data
            next_token = int(np.argmax(logits[0, -1, :]))
            generated_tokens.append(next_token)

            # Phase 2: autoregressive loop — one token at a time
            for _ in range(MAX_TOKENS - 1):
                if next_token == TOKEN_EOT:
                    break

                inputs = {
                    "input_ids":             np.array([[next_token]], dtype=np.int64),
                    "encoder_hidden_states": audio_features,
                    "beam_idx":              np.array([0], dtype=np.int32),
                }
                infer_request.infer(inputs)
                logits = infer_request.get_output_tensor(0).data
                next_token = int(np.argmax(logits[0, -1, :]))
                generated_tokens.append(next_token)

            text = self.processor.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )
            return text.strip()
    
    def transcribe(self, mel: np.ndarray):
            """Full pipeline: mel spectrogram → transcription with timing."""
            t0 = time.time()

            audio_features = self.encode(mel)
            t_encode = time.time()

            text = self.decode(audio_features)
            t_decode = time.time()

            return {
                "text":         text,
                "encode_ms":    round((t_encode - t0) * 1000),
                "decode_ms":    round((t_decode - t_encode) * 1000),
                "total_ms":     round((t_decode - t0) * 1000),
            }


        