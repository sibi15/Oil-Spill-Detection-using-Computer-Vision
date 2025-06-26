# Backend Deployment Guide for Render

## Prerequisites

1. **Render Account**: Sign up at [render.com](https://render.com)
2. **GitHub Repository**: Your code should be in a GitHub repository
3. **Cloud Storage**: You'll need to host your model files in a cloud storage service

## Step-by-Step Deployment Guide

### 1. Prepare Your Model Files

1. The model file is already hosted on Google Drive and will be downloaded automatically during deployment.
   - Model URL: [https://drive.google.com/file/d/1Q0lN2ZCRygp6iPMmnYN6eM3whVObGsWO/view](https://drive.google.com/file/d/1Q0lN2ZCRygp6iPMmnYN6eM3whVObGsWO/view)
   - The application will automatically download this model during its first startup on Render.

### 2. Update Environment Variables

In your Render dashboard, add these environment variables:

No additional environment variables are needed for model download since the URL is hardcoded in the application.
- `PORT`: 8080 (default)
- `PYTHONUNBUFFERED`: 1

### 3. Create Render Web Service

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New +" -> "Web Service"
3. Connect your GitHub repository
4. Select the branch you want to deploy
5. Set the following build settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Environment Variables**: Add the ones listed above

### 4. Verify Deployment

1. Wait for the build to complete
2. Check the logs for any errors
3. Test the API endpoint:
   ```bash
   curl -X POST -H "Content-Type: multipart/form-data" \
   -F "file=@your_test_image.jpg" \
   https://your-app-name.onrender.com/predict
   ```

## Important Notes

1. **Model Download**: The first request will download the model from your storage URL
2. **Cache**: The downloaded model will be cached for subsequent requests
3. **Memory**: Ensure your Render service has enough memory to load the model
4. **Timeout**: Consider increasing the request timeout in Render settings

## Troubleshooting

1. **Model Not Downloading**:
   - Check if `MODEL_DOWNLOAD_URL` is correct
   - Verify the URL is accessible
   - Check Render logs for download errors

2. **Memory Issues**:
   - Increase the Render service memory allocation
   - Consider compressing the model file
   - Use model quantization if possible

3. **Slow First Request**:
   - This is normal as it needs to download the model
   - Subsequent requests will be faster

## Error Handling

The application has built-in error handling for:
- Model download failures
- Prediction errors
- Image processing errors
- Missing environment variables

All errors are logged and returned as appropriate HTTP responses.
