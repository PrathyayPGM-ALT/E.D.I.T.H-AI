import requests
import base64
import io

class BrainClient:
    def __init__(self, server_url="http://localhost:5050"):
        self.url = server_url.rstrip('/')
        self._check_connection()

    def _check_connection(self):
        try:
            r = requests.get(f"{self.url}/health", timeout=5)
            if r.status_code == 200:
                print(f"[Brain] Connected to local AI at {self.url}")
                return True
        except Exception:
            pass
        print(f"[Brain] WARNING: Cannot reach {self.url}")
        print(f"[Brain] Start the brain server: cd edith_brain && python brain_server.py")
        return False

    def ask(self, question):
        try:
            r = requests.post(f"{self.url}/ask", json={"question": question}, timeout=60)
            return r.json().get("answer", "No response.")
        except Exception as e:
            return f"Brain error: {e}"

    def vision(self, frame, prompt="Describe what you see."):
        try:
            import cv2
            _, buffer = cv2.imencode('.jpg', frame)
            img_b64 = base64.b64encode(buffer).decode('utf-8')

            r = requests.post(f"{self.url}/vision",
                              json={"image": img_b64, "prompt": prompt}, timeout=120)
            return r.json().get("answer", "No response.")
        except Exception as e:
            return f"Vision error: {e}"

    def transcribe(self, audio_data):
        try:
            wav_bytes = audio_data.get_wav_data(convert_rate=16000, convert_width=2)
            files = {"audio": ("audio.wav", io.BytesIO(wav_bytes), "audio/wav")}
            r = requests.post(f"{self.url}/transcribe", files=files, timeout=30)
            text = r.json().get("text", "")
            return text if text else None
        except Exception as e:
            print(f"[Brain STT Error] {e}")
            return None

    def clear(self):
        try:
            requests.post(f"{self.url}/clear", timeout=5)
        except Exception:
            pass
