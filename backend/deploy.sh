#!/bin/bash

# Set Python version explicitly
export PYTHON_VERSION=3.8.13

# Set environment variables
export PORT=${PORT:-8080}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

# Create necessary directories first
mkdir -p models
mkdir -p /tmp/uploads
mkdir -p /tmp/results
mkdir -p /tmp/labels

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip first
pip install --upgrade pip --no-cache-dir

# Install dependencies with --no-cache-dir to save space
pip install --no-cache-dir -r requirements.txt

# Download models in a separate step
python download_models.py

# Set permissions for gunicorn
chmod +x venv/bin/gunicorn
mkdir -p /tmp/models
mkdir -p /tmp/uploads
mkdir -p /tmp/results

# Start the application using the virtual environment's Python
exec venv/bin/gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 300
