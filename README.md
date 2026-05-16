# E.D.I.T.H. (Local Brain Edition)

> **E**ven **D**ead, **I**'m **T**he **H**ero
> *(or "Extremely Dumb Idea That Happened" — depending on the day)*

A pair of AI smart glasses inspired by the ones Tony Stark gave Peter Parker right before Peter accidentally ordered a drone strike on his classmate. We promise ours are less murder-y.

**This is the no-API-key version.** Everything runs on YOUR machine. No cloud. No API keys. No "your data is being trained on by a guy in a hoodie."

## What does it actually do?

You wear glasses. You say things. The glasses do things. Magic*.

*magic = a Raspberry Pi, a microphone duct-taped to your face, and four open-source AI models tag-teaming each other while you take the credit.

### Real features:
-  **Solves problems** written on paper, whiteboards, blackboards, the back of your hand, etc.
-  **Answers questions** about literally anything (powered by Llama 3.1 8B, locally)
-  **Sees what you see** and tells you what it is (LLaVA 7B, locally)
-  **Reads text** out loud (OCR + TTS)
-  **Translates** anything you point at, into any language
-  **Takes photos and videos** because we live in a society
-  **Remembers stuff** so you don't have to ("EDITH, remember where I parked")
-  **Sets reminders** ("EDITH, remind me to stop building dumb projects in 5 minutes")
-  **Monitors your heart rate** (optional, if you sprung for the MAX30102)
-  **Bluetooth speaker support** so your friends can hear EDITH judge your math homework

## Hardware

| Part | Purpose | Approximate cost |
|---|---|---|
| Raspberry Pi Zero 2W | The brain | $15 |
| Pi Camera Module | The eye | $10 |
| INMP441 I2S Mic | The ear | $5 |
| TTP223 Touch Sensor | The "talk to me" button | $2 |
| Bluetooth speaker | The mouth | Whatever you've got |
| Optional: MAX30102 | The doctor | $5 |
| A pair of glasses frames | The actual glasses | Free (Goodwill) |
| Sanity | n/a | Sold separately |

**Total:** ~$40 + your dignity

**Note:** the Pi Zero 2W can't actually run the AI models (only 512MB RAM). You either run the brain on your PC and let the Pi connect to it over WiFi, OR upgrade to a Pi 5 8GB. See the two modes below.

## How it works

1. You **double-tap** the side of the glasses
2. *beep* (the universal signal that something is listening)
3. You say a thing
4. EDITH transcribes it via **faster-whisper** (local, on your machine)
5. **LLaVA 7B** vision model looks at your camera feed
6. **Llama 3.1 8B** thinks really hard about it
7. **Edge TTS** speaks the answer through your Bluetooth speaker in a sophisticated British accent
8. You look very smart in front of your friends

Total round-trip time: depends on your CPU. With a decent NVIDIA GPU: ~3 seconds. CPU only: 10-30 seconds. Patience is a virtue.

## Setup

```bash
# Clone the repo
git clone <this repo>
cd edith_brain

# Run the setup script
chmod +x setup_brain.sh
./setup_brain.sh
# (or setup_brain.bat on Windows)
```

This installs Ollama, pulls Llama 3.1 8B (~4.7 GB) + LLaVA 7B (~4.5 GB), and installs Python deps. About 10 GB total disk, takes 20-30 minutes depending on internet. And your Sanity

Then:

```bash
# All-in-one mode (runs on your PC):
python edith_standalone.py

# Or split mode — brain on PC, body on Pi:
python brain_server.py        # on your PC
python ../edith_glasses.py    # on the Pi (points at brain_server)
```

## Modes

### Tap mode (default)
- Glasses are quiet. They mind their business.
- Double-tap → conversation starts → ask anything
- Single-tap during convo → follow-up question
- No tap for 15s → "okay, talk to you later" → back to sleep

### Always-on mode
- Set `TAP_MODE = False` in config
- Say "EDITH" any time and she perks up
- Marginally more cyberpunk
- Drains the battery faster

### Standalone mode (no Pi at all)
- Just run `edith_standalone.py` on your laptop
- Uses your laptop's webcam, mic, and speakers
- Great for testing or if you want a desktop AI assistant without all the soldering

### Brain server mode (Pi glasses + PC brain)
- PC runs the heavy AI (`brain_server.py`) (BOOOOM)
- Pi glasses are dumb terminals that send audio/images over WiFi
- The only way to use real glasses since the Pi Zero can't fit a 7B model in 512MB RAM

## Files in this repo

| File | What it is |
|---|---|
| `edith_standalone.py` | All-in-one. Run this on a PC with webcam + mic. |
| `brain_server.py` | Flask server that hosts the AI for a remote Pi |
| `brain_client.py` | Client that the Pi uses to talk to brain_server |
| `edith_ai_local.py` | The EDITH class wired up to the local brain |
| `llm_engine.py` | Llama 3.1 wrapper (text Q&A) |
| `vision_engine.py` | LLaVA wrapper (image understanding) |
| `stt_engine.py` | faster-whisper wrapper (speech-to-text) |
| `tts_engine.py` | Edge TTS wrapper (text-to-speech) |
| `setup_brain.sh` / `.bat` | One-command install |
| `requirements.txt` | The usual |

## FAQ

**Q: Can I summon a drone strike with this?**
A: No. Please stop asking.

**Q: Does it work offline?**
A: Yes, except Edge TTS streams from Microsoft. If you want truly offline, swap it for [Piper TTS](https://github.com/rhasspy/piper). Then you're truly free of Big Tech, which makes the FBI very interested in you.

**Q: Why is it called EDITH?**
A: Same reason you're reading this README — Spider-Man: Far From Home.

**Q: What if my friends laugh at me for wearing computer glasses?**
A: They will. That's the price of being from the future.

**Q: Will it pass the Turing test?**
A: It will pass the "is my friend's project actually working" test, which is way harder.

**Q: Why did you remove the OLED display?**
A: Trying to mount a transparent OLED in a $5 pair of glasses turned out to be a "research project" not a "weekend hack." Audio-only is honestly better — you don't look like a deranged Google Glass beta tester.

**Q: Is this safe to use while driving?**
A: Why would you ask me that.

**Q: It's slow on my computer.**
A: Welcome to local AI without a GPU. Either buy an RTX 3060 or quantize down to a smaller model (`phi3:mini`, `moondream`, `whisper-tiny`). Or just be patient. Like the monks.

**Q: Can I use a different model?**
A: Yes. Edit `llm_engine.py` and `vision_engine.py` — swap `llama3.1:8b` for any Ollama model. Want `qwen2.5:14b`? Go nuts. Want `phi3:mini` for a Pi? Even better. Want `gpt-oss:120b`? Lol, hope your RAM stick is structural.

## Known issues

- Sometimes Whisper transcribes "EDITH" as "idiot." We took it personally and made "idiot" a valid wake word.
- The British TTS voice is *very* judgmental. We consider this a feature.
- If you say "EDITH, shut up" she will, in fact, shut up. This was our most-requested feature during development.
- Ollama eats RAM like it's free. It is free, but your RAM isn't.
- First-time inference is slow because the models have to load. After that, it's just regular-slow.

## Credits

- **Marvel** for the inspiration
- **Meta** for open-sourcing Llama 3.1
- **The LLaVA team** for free vision
- **Ollama** for making local LLMs not painful
- **OpenAI** for Whisper (faster-whisper is the optimized fork)
- **Microsoft Edge TTS** for the free British voice (our favorite kind)
- **The poor souls who built `webrtcvad`** so the mic stops listening when we stop talking
- **My friend** who's making the actual glasses while I write Python and act like I helped

## License

MIT. Do whatever. Just don't blame me when you set yourself on fire.

---

*"Even Dead, I'm The Hero" is a stupid backronym Tony Stark came up with at 3 AM. We honor that legacy.*
