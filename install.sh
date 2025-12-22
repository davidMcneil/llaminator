#!/bin/bash

set -e

echo "Checking for ollama..."
if command -v ollama >/dev/null 2>&1; then
    echo "ollama is already installed, skipping installation"
else
    echo "Installing ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo "Downloading ollama models..."
ollama pull moondream:1.8b
ollama pull gemma3:4b

echo "Setting up Python virtual environment..."
python3 -m venv venv

echo "Installing Python dependencies..."
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

echo "Generating SSL certificates for HTTPS..."
if [ ! -f cert.pem ] || [ ! -f key.pem ]; then
    ./venv/bin/python3 certs.py
else
    echo "SSL certificates already exist, skipping generation"
fi

echo "Installation complete"
echo "You can now run ./run.sh"
