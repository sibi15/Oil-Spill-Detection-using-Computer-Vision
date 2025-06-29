#!/bin/bash

# Exit on error
set -e

# Check if we're in the correct directory
if [ ! -f "app.py" ]; then
    echo "Error: This script must be run from the backend directory"
    exit 1
fi

# Verify Python environment
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Set up caching for pip packages
export PIP_CACHE_DIR=/tmp/pip-cache
mkdir -p $PIP_CACHE_DIR

# Install all packages in one command
pip install --no-cache-dir -r requirements.txt

# Verify model file exists
MODEL_FILE="models/sar_model_converted.tflite"
if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: Model file not found at $MODEL_FILE"
    exit 1
fi

# Verify we can read the model file
if ! test -r "$MODEL_FILE"; then
    echo "Error: Model file is not readable"
    exit 1
fi

# Verify the model file size is reasonable
MIN_MODEL_SIZE=1000000  # 1MB minimum
if [ $(stat -c%s "$MODEL_FILE") -lt $MIN_MODEL_SIZE ]; then
    echo "Error: Model file appears to be corrupted or empty"
    exit 1
fi

echo "Model file successfully verified in models directory"

# Verify dependencies
if ! python3 -c "import tensorflow" &> /dev/null; then
    echo "Error: TensorFlow is not installed"
    exit 1
fi

# Create necessary directories
mkdir -p uploads results models labels

# Verify directory structure
REQUIRED_DIRS=("uploads" "results" "models" "labels")
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "Error: Required directory $dir not found"
        exit 1
    fi
    chmod 755 "$dir"
done

# Set permissions
chmod -R 755 uploads results models labels

# Clean up cache
if [ -n "$PIP_CACHE_DIR" ]; then
    rm -rf "$PIP_CACHE_DIR"
fi

echo "Deployment verification completed successfully"
