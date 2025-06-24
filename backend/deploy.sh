#!/bin/bash

# Set Python version explicitly
export PYTHON_VERSION=3.8.13

# Set environment variables
export PORT=${PORT:-8080}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies with --break-system-packages
pip install --break-system-packages --no-cache-dir -r requirements.txt

# Create necessary directories
mkdir -p /tmp/models
mkdir -p /tmp/uploads
mkdir -p /tmp/results

# Install gunicorn in virtual environment
pip install gunicorn

# Start the application using the virtual environment's Python
exec venv/bin/gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 300
