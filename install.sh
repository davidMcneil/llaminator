#!/bin/bash

set -e

echo "Installing ollama..."
curl -fsSL https://ollama.com/install.sh | sh

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
    openssl req -x509 -newkey rsa:4096 -nodes \
        -keyout key.pem \
        -out cert.pem \
        -days 365 \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
    chmod 600 key.pem
    chmod 644 cert.pem
    echo "Self-signed certificate generated (for testing only - browsers will show security warning)"
else
    echo "SSL certificates already exist, skipping generation"
fi

echo "Installation complete"
echo "You can now run ./run.sh"
