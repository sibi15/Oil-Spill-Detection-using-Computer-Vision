#!/bin/bash

# Set up caching for pip packages
export PIP_CACHE_DIR=/tmp/pip-cache
mkdir -p $PIP_CACHE_DIR

# Install essential packages
pip install --no-cache-dir flask==2.3.3 flask-cors==4.0.0 gunicorn==21.2.0

# Install TensorFlow packages
pip install --no-cache-dir tensorflow==2.13.0

# Install remaining packages
pip install --no-cache-dir -r requirements.txt

# Create necessary directories
mkdir -p uploads results models labels

# Copy model file from project root to both locations
# First copy to backend models directory
cp ../models/sar_model.tflite models/

# Also copy to project root models directory (for relative path)
cp ../models/sar_model.tflite ../../models/

# Set permissions
chmod -R 755 uploads results models labels

# Clean up cache
rm -rf $PIP_CACHE_DIR
