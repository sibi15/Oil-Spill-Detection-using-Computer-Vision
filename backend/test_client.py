import requests
import json

url = "http://localhost:5001/predict"

# Prepare the files and data
files = {
    'file': ('img_0030.jpg', open('/Users/sibikarthik/Downloads/Oil-Spill-Detection-using-Computer-Vision/dataset/img_0030.jpg', 'rb'))
}
data = {
    'imageType': 'sar'
}

try:
    response = requests.post(url, files=files, data=data)
    print("\n=== Response Details ===")
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {response.headers}")
    print(f"Response Text: {response.text}")
    
except Exception as e:
    print(f"\n=== Request Error ===")
    print(f"Error: {str(e)}")
