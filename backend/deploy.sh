#!/bin/bash

# Set environment variables
export PORT=${PORT:-8080}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

# Install dependencies
pip install --no-cache-dir -r requirements.txt

# Create necessary directories
mkdir -p /tmp/models
mkdir -p /tmp/uploads
mkdir -p /tmp/results

# Start the application
python -m pip install gunicorn
exec gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 300
