#!/bin/bash

# Set environment variables
export PORT=8080
export PYTHONUNBUFFERED=1

# Install dependencies
pip install -r requirements.txt

# Start the application
gunicorn app:app --bind 0.0.0.0:$PORT
