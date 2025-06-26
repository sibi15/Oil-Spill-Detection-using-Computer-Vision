#!/bin/bash

# Set up caching for pip packages
export PIP_CACHE_DIR=/tmp/pip-cache
mkdir -p $PIP_CACHE_DIR

# Install dependencies with --no-cache-dir to prevent downloading all packages again
pip install --no-cache-dir -r requirements.txt

# Create necessary directories
mkdir -p uploads results models labels

# Set permissions
chmod -R 755 uploads results models labels
