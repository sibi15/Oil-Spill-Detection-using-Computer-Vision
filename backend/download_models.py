#!/usr/bin/env python3
# Script to download model files at runtime from external storage.

import os
import requests
import zipfile
from pathlib import Path

# Model URLs - Models uploaded to a cloud storage service

MODEL_URLS = {
    'infrared_model.keras': 'https://drive.google.com/uc?export=download&id=1azpgoH2M52HQtjNj3V_aeLzy_X5xgsXu',
    'sar_model.keras': 'https://drive.google.com/uc?export=download&id=1le5uHObuGbiQKyw_r8p9JgY_6eko-9h1'
}

def download_model(model_name: str, url: str, models_dir: str = 'models'):
    # Download a model file from URL
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, model_name)
    
    if os.path.exists(model_path):
        print(f"Model {model_name} already exists, skipping download.")
        return model_path
    
    print(f"Downloading {model_name} from {url}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Successfully downloaded {model_name}")
        return model_path
        
    except Exception as e:
        print(f"Error downloading {model_name}: {e}")
        return None

def download_all_models():
    # Download all required models
    for model_name, url in MODEL_URLS.items():
        if url != f'YOUR_{model_name.upper().replace(".", "_")}_URL_HERE':
            download_model(model_name, url)
        else:
            print(f"Please set the URL for {model_name} in MODEL_URLS")

if __name__ == "__main__":
    download_all_models() 