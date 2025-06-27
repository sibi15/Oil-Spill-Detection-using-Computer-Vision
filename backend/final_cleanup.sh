#!/bin/bash

# Remove unnecessary model files
echo "Cleaning up model files..."
rm -f models/sar_model.h5
rm -f models/sar_model.tflite
rm -rf models/saved_model

# Remove macOS metadata
echo "Removing .DS_Store files..."
find . -name ".DS_Store" -delete

# Stage and commit changes
echo "Staging and committing changes..."
git add .
git commit -m "cleanup: Remove unnecessary model files and metadata"
git push

echo "Final cleanup complete!"
