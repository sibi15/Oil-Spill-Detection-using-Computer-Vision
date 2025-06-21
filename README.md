# Oil Spill Detection using Computer Vision

A full-stack web application for detecting oil spills in satellite imagery using deep learning models.

## Features

- **Multiple Image Types**: Support for SAR (Synthetic Aperture Radar) and Infrared imagery
- **Real-time Analysis**: Upload images and get instant oil spill detection results
- **Advanced Metrics**: IoU, Dice coefficient, precision, recall, and density analysis
- **Visual Results**: Processed images with density maps and statistical visualizations
- **Modern UI**: Built with React, TypeScript, and Tailwind CSS

## Tech Stack

### Frontend
- React 18 with TypeScript
- Vite for build tooling
- Tailwind CSS for styling
- Axios for API calls
- Lucide React for icons

### Backend
- Flask (Python)
- TensorFlow for deep learning
- OpenCV for image processing
- Matplotlib for visualizations
- CORS enabled for cross-origin requests

## Deployment on Vercel

### Prerequisites

1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
2. **Model Hosting**: Upload your model files to a cloud storage service (Google Drive, AWS S3, etc.)
3. **Vercel CLI** (optional): `npm i -g vercel`

### Step 1: Prepare Your Models

Since Vercel has a 50MB file size limit and your models are 400MB+, you need to host them externally:

1. Upload your model files to a cloud storage service:
   - `infrared_model.keras`
   - `sar_model.keras`

2. Get direct download URLs for each model

3. Update the model URLs in `backend/app.py`:
   ```python
   MODEL_URLS = {
       'infrared_model.keras': 'YOUR_INFRARED_MODEL_URL_HERE',
       'sar_model.keras': 'YOUR_SAR_MODEL_URL_HERE'
   }
   ```

### Step 2: Configure Environment Variables

In your Vercel dashboard, add these environment variables:

- `VITE_API_URL`: Your Vercel app URL (e.g., `https://your-app.vercel.app/api`)

### Step 3: Deploy to Vercel

#### Option A: Using Vercel Dashboard
1. Push your code to GitHub
2. Connect your repository to Vercel
3. Configure build settings:
   - **Framework Preset**: Vite
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
   - **Install Command**: `npm install`

#### Option B: Using Vercel CLI
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Follow the prompts to configure your deployment
```

### Step 4: Verify Deployment

1. Check that your frontend is accessible at your Vercel URL
2. Test the API endpoint at `/api/predict`
3. Verify that models are downloaded correctly on first use

## Local Development

### Frontend
```bash
npm install
npm run dev
```

### Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Combined Development
```bash
npm run dev  # This runs both frontend and backend concurrently
```

## Project Structure

```
├── src/                    # Frontend React code
│   ├── components/         # React components
│   ├── utils/             # Utility functions
│   └── config.ts          # Configuration
├── backend/               # Python Flask backend
│   ├── app.py            # Main Flask application
│   ├── requirements.txt  # Python dependencies
│   ├── models/           # Model files (not in git)
│   └── download_models.py # Model download script
├── public/               # Static assets
├── vercel.json          # Vercel configuration
└── package.json         # Node.js dependencies
```

## API Endpoints

### POST /api/predict
Upload an image for oil spill detection.

**Request:**
- `file`: Image file (multipart/form-data)
- `imageType`: Type of image ('sar' or 'infrared')

**Response:**
```json
{
  "processed_image": "base64_encoded_image",
  "density_graph": "base64_encoded_graph",
  "metrics": {
    "dice": 0.85,
    "iou": 0.74,
    "spill_area": 15.2,
    "precision": 0.89,
    "recall": 0.82,
    "mean_intensity": 45.6,
    "standard_deviation": 12.3
  }
}
```

## Troubleshooting

### Common Issues

1. **Models not downloading**: Check your model URLs in `backend/app.py`
2. **CORS errors**: Ensure CORS is properly configured in the Flask app
3. **Large file uploads**: Vercel has limits on request body size
4. **Memory issues**: The serverless function has memory limits

### Performance Optimization

1. **Model caching**: Models are downloaded once and cached
2. **Image compression**: Consider compressing uploaded images
3. **CDN**: Use a CDN for serving model files

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License. 