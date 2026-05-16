import cv2
import os
import json
import time
import re
import threading
from datetime import datetime, timedelta

from brain_client import BrainClient
from tts_engine import TTSEngine

CAPTURE_PATH = "../captures/"

class EdithAI:
    def __init__(self, brain_url="http://localhost:5050"):
        self.brain = BrainClient(brain_url)
        self.tts = TTSEngine()

        self.camera = None
        self.camera_active = False
        self.last_frame = None

        self.is_recording = False
        self.video_writer = None
        self.record_thread = None
        self._record_filename = ""

        self.is_speaking = False

        self.memory_file = os.path.join(CAPTURE_PATH, "edith_memory.json")
        self.memories = self._load_memories()

        self.reminders = []
        self.on_reminder = None
        threading.Thread(target=self._reminder_loop, daemon=True).start()

        os.makedirs(CAPTURE_PATH, exist_ok=True)

    def start_camera(self):
        if self.camera is None or not self.camera.isOpened():
            self.camera = cv2.VideoCapture(0)
            self.camera_active = True
        return self.camera.isOpened()

    def stop_camera(self):
        if self.camera:
            self.camera_active = False
            self.camera.release()
            self.camera = None

    def get_frame(self):
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                self.last_frame = frame
                return frame
        return None

    def take_picture(self):
        frame = self.get_frame()
        if frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(CAPTURE_PATH, f"edith_capture_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            return filename
        return None

    def start_recording(self):
        if self.is_recording:
            return "Already recording."
        frame = self.get_frame()
        if frame is None:
            return "Camera not available."

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(CAPTURE_PATH, f"edith_video_{timestamp}.avi")
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (w, h))
        self.is_recording = True
        self._record_filename = filename

        def _record():
            while self.is_recording and self.camera and self.camera.isOpened():
                ret, f = self.camera.read()
                if ret and self.video_writer:
                    self.video_writer.write(f)
                    self.last_frame = f
                time.sleep(0.05)

        self.record_thread = threading.Thread(target=_record, daemon=True)
        self.record_thread.start()
        return filename

    def stop_recording(self):
        if not self.is_recording:
            return "Not recording."
        self.is_recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        return self._record_filename

    def ask(self, question, include_image=False):
        if include_image:
            frame = self.get_frame()
            if frame is not None:
                return self.brain.vision(frame, question)
        return self.brain.ask(question)

    def solve_problem(self):
        frame = self.get_frame()
        if frame is None:
            return "Camera not available."
        return self.brain.vision(
            frame,
            "Look at this image. Identify any problem (math, physics, chemistry, code, etc.) "
            "written on the paper/board/screen. Solve it step by step. "
            "If you can't identify a clear problem, describe what you see."
        )

    def read_text(self):
        frame = self.get_frame()
        if frame is None:
            return "Camera not available."
        return self.brain.vision(
            frame,
            "Read ALL the text you can see in this image. "
            "Reproduce it exactly as written. If there's no readable text, say so."
        )

    def translate(self, target_language="English"):
        frame = self.get_frame()
        if frame is None:
            return "Camera not available."
        return self.brain.vision(
            frame,
            f"Read any text visible in this image and translate it to {target_language}. "
            "Show the original text and the translation."
        )

    def translate_speech(self, spoken_text, target_language):
        return self.brain.ask(
            f'Translate the following to {target_language}: "{spoken_text}"\n'
            "Give only the translation, nothing else."
        )

    def summarize_view(self):
        frame = self.get_frame()
        if frame is None:
            return "Camera not available."
        return self.brain.vision(
            frame,
            "Look at this image. If there's text, give a concise summary. "
            "If it's a scene, summarize what's happening."
        )

    def scan_qr(self):
        frame = self.get_frame()
        if frame is None:
            return "Camera not available."

        detector = cv2.QRCodeDetector()
        data, vertices, _ = detector.detectAndDecode(frame)
        if data:
            return f"QR Code detected:\n{data}"

        return self.brain.vision(
            frame,
            "Is there a QR code, barcode, or scannable code? "
            "If yes, decode it. If not, say 'No code detected.'"
        )

    def _load_memories(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save_memories(self):
        with open(self.memory_file, 'w') as f:
            json.dump(self.memories, f, indent=2)

    def remember(self, note, include_image=False):
        memory = {"timestamp": datetime.now().isoformat(), "note": note, "image": None}
        if include_image:
            filename = self.take_picture()
            if filename:
                memory["image"] = filename
        self.memories.append(memory)
        self._save_memories()
        return f"Remembered: {note}" + (f" (with photo)" if memory['image'] else "")

    def recall(self, query=""):
        if not self.memories:
            return "No memories saved yet."
        if not query:
            recent = self.memories[-5:]
            lines = ["Recent memories:"]
            for m in reversed(recent):
                t = m['timestamp'][:16].replace('T', ' ')
                lines.append(f"  [{t}] {m['note']}")
            return "\n".join(lines)

        memory_text = "\n".join(f"[{m['timestamp'][:16]}] {m['note']}" for m in self.memories)
        return self.brain.ask(
            f"Here are saved memories:\n{memory_text}\n\nSearch for: {query}\nReturn relevant ones."
        )

    def forget_all(self):
        self.memories = []
        self._save_memories()
        return "All memories cleared."

    def set_reminder(self, text, minutes):
        trigger_time = datetime.now() + timedelta(minutes=minutes)
        self.reminders.append({"text": text, "trigger_time": trigger_time, "fired": False})
        return f"Reminder set: '{text}' in {minutes} minute{'s' if minutes != 1 else ''}."

    def parse_reminder(self, command):
        patterns = [
            (r'(\d+)\s*minute', lambda m: int(m.group(1))),
            (r'(\d+)\s*min', lambda m: int(m.group(1))),
            (r'(\d+)\s*hour', lambda m: int(m.group(1)) * 60),
            (r'(\d+)\s*second', lambda m: max(1, int(m.group(1)) // 60)),
            (r'half\s*(?:an?\s*)?hour', lambda m: 30),
        ]
        minutes = None
        for pattern, extractor in patterns:
            match = re.search(pattern, command, re.IGNORECASE)
            if match:
                minutes = extractor(match)
                break
        if minutes is None:
            return None
        msg = command
        for strip in ['remind me', 'set a reminder', 'set reminder', 'reminder',
                       'in \d+ minutes?', 'in \d+ mins?', 'in \d+ hours?',
                       'in \d+ seconds?', 'in half an hour', 'to ', 'that ']:
            msg = re.sub(strip, '', msg, flags=re.IGNORECASE).strip()
        if not msg:
            msg = "Reminder"
        return msg, minutes

    def list_reminders(self):
        active = [r for r in self.reminders if not r['fired']]
        if not active:
            return "No active reminders."
        lines = ["Active reminders:"]
        now = datetime.now()
        for r in active:
            remaining = (r['trigger_time'] - now).total_seconds() / 60
            if remaining > 0:
                lines.append(f"  '{r['text']}' — {remaining:.0f} min remaining")
        return "\n".join(lines)

    def _reminder_loop(self):
        while True:
            now = datetime.now()
            for r in self.reminders:
                if not r['fired'] and now >= r['trigger_time']:
                    r['fired'] = True
                    if self.on_reminder:
                        self.on_reminder(r['text'])
            time.sleep(5)

    def get_time(self):
        return datetime.now().strftime("It's %I:%M %p, %A %B %d, %Y.")

    def get_system_info(self):
        import platform
        return (
            f"System: {platform.system()} {platform.release()}\n"
            f"Machine: {platform.machine()}\n"
            f"Python: {platform.python_version()}\n"
            f"Time: {self.get_time()}\n"
            f"Memories: {len(self.memories)}\n"
            f"Reminders: {len([r for r in self.reminders if not r['fired']])}\n"
            f"AI: Local (Ollama + Whisper)"
        )

    def speak(self, text):
        self.is_speaking = True
        self.tts.speak(text)
        self.is_speaking = False

    def stop_speaking(self):
        self.is_speaking = False
        self.tts.stop()

    def transcribe(self, audio_data):
        return self.brain.transcribe(audio_data)

    def shutdown(self):
        if self.is_recording:
            self.stop_recording()
        self.stop_camera()
        self.tts.shutdown()

if __name__ == "__main__":
    edith = EdithAI()
    print("E.D.I.T.H. (Local AI) initialized.")
    response = edith.ask("Hello EDITH, what can you do?")
    print(f"EDITH: {response}")
    edith.shutdown()
