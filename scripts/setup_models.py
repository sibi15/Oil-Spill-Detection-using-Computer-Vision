#!/usr/bin/env python3
# Script to upload model files to cloud storage (AWS, GitHub assets, or GDrive) and generate download URLs

import os
import sys
from pathlib import Path

def check_models_exist():
    #Check if model files exist in the backend/models directory
    models_dir = Path("backend/models")
    models = ["infrared_model.keras", "sar_model.keras"]
    
    print("Checking for model files...")
    existing_models = []
    
    for model in models:
        model_path = models_dir / model
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"✅ {model} found ({size_mb:.1f} MB)")
            existing_models.append(model)
        else:
            print(f"❌ {model} not found")
    
    return existing_models

def google_drive_instructions():
    print("\n" + "="*50)
    print("GOOGLE DRIVE UPLOAD INSTRUCTIONS")
    print("="*50)
    print("1. Go to drive.google.com")
    print("2. Create a new folder for your models")
    print("3. Upload your .keras model files")
    print("4. Right-click each file → 'Share' → 'Copy link'")
    print("5. Replace 'https://drive.google.com/file/d/FILE_ID/view?usp=sharing'")
    print("   with 'https://drive.google.com/uc?export=download&id=FILE_ID'")
    print("6. Update MODEL_URLS in backend/app.py")

def aws_s3_instructions():
    print("\n" + "="*50)
    print("AWS S3 UPLOAD INSTRUCTIONS")
    print("="*50)
    print("1. Create an S3 bucket")
    print("2. Upload your .keras model files")
    print("3. Make files publicly accessible")
    print("4. Get the public URL for each file")
    print("5. Update MODEL_URLS in backend/app.py")
    print("\nExample URL format:")
    print("https://your-bucket.s3.amazonaws.com/infrared_model.keras")

def github_releases_instructions():
    print("\n" + "="*50)
    print("GITHUB RELEASES UPLOAD INSTRUCTIONS")
    print("="*50)
    print("1. Create a new repository for your models")
    print("2. Create a new release")
    print("3. Upload your .keras model files as release assets")
    print("4. Get the download URL for each file")
    print("5. Update MODEL_URLS in backend/app.py")
    print("\nExample URL format:")
    print("https://github.com/username/repo/releases/download/v1.0/infrared_model.keras")

def update_model_urls_template():
    print("\n" + "="*50)
    print("UPDATE MODEL_URLS TEMPLATE")
    print("="*50)
    print("In backend/app.py, update the MODEL_URLS dictionary:")
    print()
    print("MODEL_URLS = {")
    print("    'infrared_model.keras': 'YOUR_INFRARED_MODEL_URL_HERE',")
    print("    'sar_model.keras': 'YOUR_SAR_MODEL_URL_HERE'")
    print("}")
    print()
    print("Replace the placeholder URLs with your actual download URLs.")

def main():
    print("Oil Spill Detection - Model Setup Guide")
    print("="*50)
    
    # Check for existing models
    existing_models = check_models_exist()
    
    if not existing_models:
        print("\n❌ No model files found!")
        print("Please ensure your .keras model files are in the backend/models/ directory.")
        return
    
    print(f"\n✅ Found {len(existing_models)} model file(s)")
    
    # Show hosting options
    print("\nChoose a hosting option for your models:")
    print("1. Google Drive (Free, easy setup)")
    print("2. AWS S3 (Paid, more reliable)")
    print("3. GitHub Releases (Free, good for versioning)")
    print("4. Show all instructions")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        google_drive_instructions()
    elif choice == "2":
        aws_s3_instructions()
    elif choice == "3":
        github_releases_instructions()
    elif choice == "4":
        google_drive_instructions()
        aws_s3_instructions()
        github_releases_instructions()
    else:
        print("Invalid choice. Showing all options:")
        google_drive_instructions()
        aws_s3_instructions()
        github_releases_instructions()
    
    update_model_urls_template()

if __name__ == "__main__":
    main() 