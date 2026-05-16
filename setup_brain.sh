#!/bin/bash
set -e

echo "=========================================="
echo "  E.D.I.T.H. LOCAL AI BRAIN SETUP"
echo "=========================================="

echo "[1/4] Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

echo "[2/4] Pulling LLM model (llama3.1:8b)..."
ollama pull llama3.1:8b

echo "[3/4] Pulling vision model (llava:7b)..."
ollama pull llava:7b

echo "[4/4] Installing Python packages..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "  Done. Run: python edith_standalone.py"
echo "=========================================="
