import io
import numpy as np
from faster_whisper import WhisperModel

class STTEngine:
    def __init__(self, model_size="base.en", device="cpu"):
        self.model = WhisperModel(model_size, device=device, compute_type="int8")

    def transcribe_file(self, audio_path):
        try:
            segments, info = self.model.transcribe(audio_path, language="en")
            text = " ".join(segment.text for segment in segments).strip()
            return text if text else None
        except Exception as e:
            print(f"[STT Error] {e}")
            return None

    def transcribe_audio_data(self, audio_data):
        try:
            wav_bytes = audio_data.get_wav_data(convert_rate=16000, convert_width=2)
            wav_buffer = io.BytesIO(wav_bytes)
            wav_buffer.name = "audio.wav"

            segments, info = self.model.transcribe(wav_buffer, language="en")
            text = " ".join(segment.text for segment in segments).strip()
            return text if text else None
        except Exception as e:
            print(f"[STT Error] {e}")
            return None

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python stt_engine.py <audio_file.wav>")
        sys.exit(1)
    stt = STTEngine()
    result = stt.transcribe_file(sys.argv[1])
    print(f"Transcription: {result}")
