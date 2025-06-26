#!/bin/bash

# Set up caching for pip packages
export PIP_CACHE_DIR=/tmp/pip-cache
mkdir -p $PIP_CACHE_DIR

# Set environment variables
export PORT=8080
export SERVER_NAME=oil-spill-backend.onrender.com

# Install essential packages first
pip install --no-cache-dir flask==2.3.3 flask-cors==4.0.0 gunicorn==21.2.0

# Install TensorFlow packages separately
pip install --no-cache-dir tensorflow==2.13.0

# Install remaining packages
pip install --no-cache-dir -r requirements.txt

# Create necessary directories
mkdir -p uploads results models labels

# Set permissions
chmod -R 755 uploads results models labels

# Ensure Python files are executable
chmod +x app.py

# Clean up cache
rm -rf $PIP_CACHE_DIR
