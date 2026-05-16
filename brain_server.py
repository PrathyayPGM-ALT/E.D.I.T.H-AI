import os
import sys
import base64
import tempfile
from flask import Flask, request, jsonify

from llm_engine import LLMEngine
from vision_engine import VisionEngine
from stt_engine import STTEngine

app = Flask(__name__)

print("[BRAIN] Loading AI models...")
llm = LLMEngine(model="llama3.1:8b")
vision = VisionEngine(model="llava:7b")
stt = STTEngine(model_size="base.en", device="cpu")
print("[BRAIN] All models loaded.")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "online", "models": ["llm", "vision", "stt"]})

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '')
    if not question:
        return jsonify({"error": "No question provided"}), 400
    answer = llm.ask(question)
    return jsonify({"answer": answer})

@app.route('/vision', methods=['POST'])
def vision_analyze():
    data = request.json
    prompt = data.get('prompt', 'Describe what you see.')
    image_b64 = data.get('image', '')

    if not image_b64:
        return jsonify({"error": "No image provided"}), 400

    img_bytes = base64.b64decode(image_b64)
    tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    tmp.write(img_bytes)
    tmp.close()

    result = vision.analyze(tmp.name, prompt)
    os.unlink(tmp.name)
    return jsonify({"answer": result})

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files['audio']
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    audio_file.save(tmp.name)
    tmp.close()

    text = stt.transcribe_file(tmp.name)
    os.unlink(tmp.name)
    return jsonify({"text": text or ""})

@app.route('/clear', methods=['POST'])
def clear():
    llm.clear_history()
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5050
    print(f"[BRAIN] Server running on http://0.0.0.0:{port}")
    print(f"[BRAIN] From Pi, connect to http://<this_machine_ip>:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
