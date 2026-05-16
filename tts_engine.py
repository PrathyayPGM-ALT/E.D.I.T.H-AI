import os
import time
import asyncio
import threading
import tempfile
import edge_tts
import pygame

class TTSEngine:
    def __init__(self, voice="en-GB-SoniaNeural"):
        self.voice = voice
        self.lock = threading.Lock()
        self.is_speaking = False
        self._temp_dir = tempfile.mkdtemp()
        pygame.mixer.init()

    def speak(self, text):
        if not text:
            return
        self.is_speaking = True
        with self.lock:
            try:
                audio_file = os.path.join(self._temp_dir, "edith_speech.mp3")

                async def _generate():
                    communicate = edge_tts.Communicate(text, self.voice)
                    await communicate.save(audio_file)

                asyncio.run(_generate())

                if not self.is_speaking:
                    return

                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy() and self.is_speaking:
                    time.sleep(0.05)
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
            except Exception as e:
                print(f"[TTS Error] {e}")
            finally:
                self.is_speaking = False

    def stop(self):
        self.is_speaking = False
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass

    def shutdown(self):
        pygame.mixer.quit()

if __name__ == "__main__":
    tts = TTSEngine()
    tts.speak("Hello, I am E.D.I.T.H. Your personal AI assistant.")
    tts.shutdown()
