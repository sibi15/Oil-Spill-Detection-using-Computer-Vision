import requests
import json
import os
from PIL import Image
import numpy as np
import io

def create_test_image():
    """Create a test image with a simple pattern"""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    # Create a simple pattern (e.g., a diagonal line)
    for i in range(256):
        for j in range(256):
            if abs(i - j) < 10:  # Create a thicker diagonal line
                img[i, j, :] = [255, 255, 255]
    return img

def test_api(api_url):
    """Test the API endpoint with a test image"""
    # Create a test image
    test_image = create_test_image()
    
    # Convert image to PNG format in memory
    img_buffer = io.BytesIO()
    Image.fromarray(test_image).save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    # Prepare the files and form data
    files = {
        'file': ('test_image.png', img_buffer, 'image/png')
    }
    data = {
        'imageType': 'sar'
    }
    
    # Send the request
    print(f"Sending test request to: {api_url}")
    response = requests.post(
        api_url,
        files=files,
        data=data
    )
    
    # Print response details
    print(f"\nResponse Status Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    
    try:
        response_data = response.json()
        print("\nResponse Data:")
        print(json.dumps(response_data, indent=2))
    except json.JSONDecodeError:
        print("\nResponse is not JSON:")
        print(response.text)

if __name__ == "__main__":
    # You'll need to replace this with your actual Render URL
    api_url = "https://oil-spill-backend.onrender.com/predict"
    
    print("\n=== Testing API Endpoint ===")
    test_api(api_url)
