#!/bin/bash

# Set Python version explicitly
export PYTHON_VERSION=3.10.13

# Set environment variables
export PORT=${PORT:-8080}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

# Create necessary directories
mkdir -p models
mkdir -p /tmp/uploads
mkdir -p /tmp/results
mkdir -p /tmp/labels

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download models
python download_models.py

# Verify installation
echo "Checking dependencies..."
pip list
python -c "import tensorflow; print('TensorFlow version:', tensorflow.__version__)"
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
python -c "import matplotlib; print('Matplotlib version:', matplotlib.__version__)"

# Set executable permissions for gunicorn
chmod +x venv/bin/gunicorn
mkdir -p /tmp/models
mkdir -p /tmp/uploads
mkdir -p /tmp/results

# Start the application using the virtual environment's Python
exec venv/bin/gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 300
