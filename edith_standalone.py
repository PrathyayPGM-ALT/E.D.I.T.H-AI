import threading
import time
import re
import speech_recognition as sr
from edith_ai_local import EdithAI

SOLVE_KW = ['solve', 'problem', 'answer this', 'work this out', 'calculate',
            'what is this', 'figure this out', 'work out']
CAPTURE_KW = ['take a picture', 'take a photo', 'capture', 'snap', 'photograph',
              'take pic', 'click', 'cheese']
LOOK_KW = ['what do you see', 'look at this', 'describe what', 'identify this',
           'what is in front', 'what am i looking at', 'analyze this',
           'what is that', 'what are these', 'who is that']
READ_KW = ['read this', 'read that', 'read the text', 'what does it say',
           'what does this say', 'what does that say', 'ocr']
TRANSLATE_KW = ['translate']
SUMMARIZE_KW = ['summarize', 'summary', 'sum up', 'tldr', 'key points']
SCAN_KW = ['scan', 'qr code', 'barcode', 'qr']
REMEMBER_KW = ['remember', 'save this', 'note this', 'store this', 'keep this']
RECALL_KW = ['what did i save', 'recall', 'my memories', 'what do you remember']
FORGET_KW = ['forget everything', 'clear memory', 'clear memories', 'forget all']
REMIND_KW = ['remind me', 'set a reminder', 'set reminder', 'set a timer', 'set timer']
LIST_REMIND_KW = ['my reminders', 'list reminders', 'active reminders']
RECORD_START_KW = ['start recording', 'record video', 'start video']
RECORD_STOP_KW = ['stop recording', 'end recording', 'stop video']
TIME_KW = ['what time', 'what day', 'what date', 'what is today']
SYSINFO_KW = ['system info', 'system status', 'diagnostics']
STOP_KW = ['stop', 'shut up', 'quiet', 'silence', 'be quiet', 'hush', 'enough']

WAKE_WORDS = ['edith', 'edit', 'e.d.i.t.h', 'edithe', 'edis', 'eedith',
              'hey edith', 'idiot', 'idiph', 'edif', 'ediph', 'eadith',
              'adith', 'edeth', 'edi', 'hedit']

def has_wake_word(text):
    text_lower = text.lower()
    for w in sorted(WAKE_WORDS, key=len, reverse=True):
        pos = text_lower.find(w)
        if pos != -1:
            return True, text[pos + len(w):].strip(' ,.')
    return False, text

def match_intent(command, keywords):
    return any(kw in command for kw in keywords)

def main():
    print("========================================")
    print("  E.D.I.T.H. STANDALONE (Local AI)")
    print("  No API keys. No cloud. All local.")
    print("========================================")

    edith = EdithAI()
    edith.start_camera()

    def on_reminder(text):
        edith.stop_speaking()
        print(f"\n[REMINDER] {text}")
        edith.speak(f"Reminder: {text}")

    edith.on_reminder = on_reminder

    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("Calibrating microphone...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)

    recognizer.energy_threshold = 250
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 2.0
    recognizer.non_speaking_duration = 0.4
    recognizer.phrase_threshold = 0.2

    edith.speak("EDITH online. All systems local. Ready for your command.")
    print('Say "EDITH" followed by a command.\n')

    running = True
    is_processing = False

    while running:
        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=4, phrase_time_limit=15)

            text = edith.transcribe(audio)
            if not text:
                continue

            print(f'[Heard] "{text}"')

            detected, command_after = has_wake_word(text.lower())
            if not detected:
                continue

            if edith.is_speaking:
                edith.stop_speaking()

            command = command_after
            for filler in ['please', 'can you', 'could you', 'hey', 'yo', ',', '.']:
                command = command.replace(filler, '').strip()

            if len(command.split()) < 2:
                print("Listening for more...")
                old_pause = recognizer.pause_threshold
                recognizer.pause_threshold = 3.0
                try:
                    with mic as source:
                        audio2 = recognizer.listen(source, timeout=6, phrase_time_limit=20)
                    extra = edith.transcribe(audio2)
                    if extra:
                        _, extra_cleaned = has_wake_word(extra.lower())
                        command = (command + " " + extra_cleaned).strip()
                except sr.WaitTimeoutError:
                    pass
                finally:
                    recognizer.pause_threshold = old_pause

            if not command.strip():
                edith.speak("I'm here. What do you need?")
                continue

            print(f'[Command] "{command}"')

            if match_intent(command, ['shut down', 'shutdown', 'goodbye', 'turn off']):
                edith.speak("Goodbye.")
                break

            if match_intent(command, STOP_KW):
                edith.stop_speaking()
                continue

            if match_intent(command, TIME_KW):
                r = edith.get_time()
            elif match_intent(command, SYSINFO_KW):
                r = edith.get_system_info()
            elif match_intent(command, SOLVE_KW):
                print("Analyzing...")
                r = edith.solve_problem()
            elif match_intent(command, CAPTURE_KW):
                r = f"Saved: {edith.take_picture()}"
            elif match_intent(command, RECORD_START_KW):
                r = f"Recording: {edith.start_recording()}"
            elif match_intent(command, RECORD_STOP_KW):
                r = f"Saved: {edith.stop_recording()}"
            elif match_intent(command, READ_KW):
                print("Reading...")
                r = edith.read_text()
            elif match_intent(command, TRANSLATE_KW):
                lang = "English"
                lang_match = re.search(r'(?:to|into)\s+(\w+)', command, re.IGNORECASE)
                if lang_match:
                    lang = lang_match.group(1).capitalize()
                r = edith.translate(lang)
            elif match_intent(command, SUMMARIZE_KW):
                r = edith.summarize_view()
            elif match_intent(command, SCAN_KW):
                r = edith.scan_qr()
            elif match_intent(command, REMEMBER_KW):
                note = command
                for strip in ['remember', 'save this', 'note this', 'store this', 'keep this', 'that ', 'this ']:
                    note = re.sub(strip, '', note, flags=re.IGNORECASE).strip()
                r = edith.remember(note or "Visual memory", include_image=True)
            elif match_intent(command, RECALL_KW):
                r = edith.recall()
            elif match_intent(command, FORGET_KW):
                r = edith.forget_all()
            elif match_intent(command, REMIND_KW):
                parsed = edith.parse_reminder(command)
                if parsed:
                    msg, minutes = parsed
                    r = edith.set_reminder(msg, minutes)
                else:
                    r = "Couldn't parse time. Try: remind me in 5 minutes to do something."
            elif match_intent(command, LIST_REMIND_KW):
                r = edith.list_reminders()
            elif match_intent(command, LOOK_KW):
                r = edith.ask("Describe what you see in this image. Be concise.", include_image=True)
            else:
                print("Thinking...")
                r = edith.ask(text)

            print(f"[EDITH] {r}")
            edith.speak(r)

        except sr.WaitTimeoutError:
            continue
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[Error] {e}")
            time.sleep(0.3)

    edith.shutdown()
    print("E.D.I.T.H. shutdown complete.")

if __name__ == "__main__":
    main()
