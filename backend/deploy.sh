#!/bin/bash

# Set up caching for pip packages
export PIP_CACHE_DIR=/tmp/pip-cache
mkdir -p $PIP_CACHE_DIR

# Install all packages in one command
pip install --no-cache-dir -r requirements.txt

# Create necessary directories
mkdir -p uploads results models labels

# Verify the model file exists in backend/models directory
if [ ! -f "models/sar_model_converted.tflite" ]; then
    echo "Error: Model file not found at models/sar_model.tflite"
    exit 1
fi

# Verify we can read the model file
if ! test -r "models/sar_model.tflite"; then
    echo "Error: Model file is not readable"
    exit 1
fi

# Verify the model file size is reasonable
if [ $(stat -c%s "models/sar_model.tflite") -lt 1000 ]; then
    echo "Error: Model file appears to be corrupted or empty"
    exit 1
fi

echo "Model file successfully verified in models directory"

# Set permissions
chmod -R 755 uploads results models labels

# Clean up cache
rm -rf $PIP_CACHE_DIR
