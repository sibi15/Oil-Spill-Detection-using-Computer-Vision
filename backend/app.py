import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
from skimage.transform import resize
from PIL import Image
import io
import base64
import tensorflow as tf
from werkzeug.utils import secure_filename
import traceback  # for debug exception tracing
import requests
from dotenv import load_dotenv
import cv2
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins for development

# Configuration from environment variables
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', '/tmp/uploads')
RESULTS_FOLDER = os.getenv('RESULTS_FOLDER', '/tmp/results')
MODEL_FOLDER = os.getenv('MODEL_FOLDER', '/tmp/models')
OUTPUT_SIZE = (512, 512)
SAMPLE_SIZE = (256, 256)

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Model configuration from environment variables
MODEL_URLS = {
    'infrared_model.keras': os.getenv('GDRIVE_INFRARED_MODEL_URL', 'https://drive.google.com/uc?id=1azpgoH2M52HQtjNj3V_aeLzy_X5xgsXu'),
    'sar_model.keras': os.getenv('GDRIVE_SAR_MODEL_URL', 'https://drive.google.com/uc?id=1le5uHObuGbiQKyw_r8p9JgY_6eko-9h1')
}

# Google Drive API configuration
GDRIVE_API_KEY = os.getenv('GDRIVE_API_KEY', '')

# Server configuration
PORT = int(os.getenv('PORT', 8080))
PYTHONUNBUFFERED = os.getenv('PYTHONUNBUFFERED', '1') == '1'

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins for development

# Configuration from environment variables
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', '/tmp/uploads')
RESULTS_FOLDER = os.getenv('RESULTS_FOLDER', '/tmp/results')
MODEL_FOLDER = os.getenv('MODEL_FOLDER', '/tmp/models')
OUTPUT_SIZE = (512, 512)
SAMPLE_SIZE = (256, 256)

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Model configuration from environment variables
MODEL_URLS = {
    'infrared_model.keras': os.getenv('GDRIVE_INFRARED_MODEL_URL', 'https://drive.google.com/uc?id=1azpgoH2M52HQtjNj3V_aeLzy_X5xgsXu'),
    'sar_model.keras': os.getenv('GDRIVE_SAR_MODEL_URL', 'https://drive.google.com/uc?id=1le5uHObuGbiQKyw_r8p9JgY_6eko-9h1')
}

# Google Drive API configuration
GDRIVE_API_KEY = os.getenv('GDRIVE_API_KEY', '')

# Server configuration
PORT = int(os.getenv('PORT', 8080))
PYTHONUNBUFFERED = os.getenv('PYTHONUNBUFFERED', '1') == '1'

def download_model_if_needed(model_name: str):
    # Download model if it doesn't exist locally
    model_path = os.path.join(MODEL_FOLDER, model_name)
    
    if os.path.exists(model_path):
        logger.info(f"Model {model_name} already exists locally")
        return model_path
    
    url = MODEL_URLS.get(model_name)
    if not url:
        logger.error(f"No URL configured for model {model_name}")
        return None
    
    logger.info(f"Downloading {model_name} from {url}...")
    
    try:
        # Add headers to handle Google Drive download
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # First request to get confirmation page
        response = requests.get(url, headers=headers, allow_redirects=True)
        if response.status_code != 200:
            logger.error(f"Failed to get model download page: {response.status_code}")
            return None
            
        # Extract download link from confirmation page
        download_url = None
        if 'drive.google.com' in url:
            # For Google Drive direct download
            download_url = url.replace('drive.google.com/uc?id=', 'drive.google.com/uc?export=download&id=')
        else:
            logger.error("Unsupported download URL format")
            return None
            
        # Download the actual file
        logger.info(f"Downloading from actual URL: {download_url}")
        response = requests.get(download_url, headers=headers, stream=True)
        response.raise_for_status()
        
        # Create model directory if it doesn't exist
        os.makedirs(MODEL_FOLDER, exist_ok=True)
        
        # Write the file in chunks
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    f.flush()
        
        logger.info(f"Successfully downloaded {model_name}")
        return model_path
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error downloading {model_name}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error downloading {model_name}: {str(e)}")
        return None

def load_images(image_path, label_path=None):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, OUTPUT_SIZE)
    image = tf.image.convert_image_dtype(image, tf.float32) / 255.0

    if label_path:
        label = tf.io.read_file(label_path)
        label = tf.image.decode_png(label, channels=1)
        label = tf.image.resize(label, OUTPUT_SIZE)
        label = tf.image.convert_image_dtype(label, tf.float32)
        label = tf.cast(label > 0, tf.float32)

        image = tf.image.resize(image, SAMPLE_SIZE)
        label = tf.image.resize(label, SAMPLE_SIZE)
        return image, label

    image = tf.image.resize(image, SAMPLE_SIZE)
    return image


def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-6)


def classify_density(x):
    return x  # simple pass-through density classifier


def resize_mask(mask, target_shape):
    if mask.shape != target_shape:
        return resize(mask, target_shape, mode='constant', preserve_range=True)
    return mask


@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("Received prediction request")
        
        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            logger.error("No selected file")
            return jsonify({'error': 'No selected file'}), 400

        if file:
            logger.info(f"Processing file: {file.filename}")
            
            # Get image type from request
            image_type = request.form.get('imageType')
            if not image_type:
                logger.error("Image type not specified")
                return jsonify({'error': 'Image type not specified'}), 400

            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            file.save(filepath)
            logger.info(f"File saved to: {filepath}")

            # read raw grayscale image for plotting and density
            original_image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            logger.info("Successfully read image")

        # load processed image and optional true mask
        processed_image = load_images(filepath)
        true_mask = None
        original_label = None

        # attempt auto-load static mask
        base = os.path.splitext(filename)[0]
        lbl_static = os.path.join(UPLOAD_FOLDER, f"{base}.png")
        if os.path.exists(lbl_static):
            original_label = cv2.imread(lbl_static, cv2.IMREAD_GRAYSCALE)
            processed_image, true_mask = load_images(filepath, lbl_static)

        # override with uploaded label if provided
        label_file = request.files.get('label')
        if label_file:
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            lbl_fname = secure_filename(label_file.filename)
            lbl_path = os.path.join(UPLOAD_FOLDER, lbl_fname)
            label_file.save(lbl_path)
            original_label = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
            processed_image, true_mask = load_images(filepath, lbl_path)

        try:
            # select model path; use cross-platform path
            model_path = os.path.join(MODEL_FOLDER, f"{image_type}_model.keras")
            logger.info(f"Looking for model at: {model_path}")
            
            if not os.path.exists(model_path):
                logger.info("Model not found locally, downloading...")
                model_path = download_model_if_needed(f"{image_type}_model.keras")
                if not model_path:
                    logger.error(f"Failed to download model for {image_type}")
                    return jsonify({'error': f'Model for {image_type} not found'}), 404

            try:
                # Load model with optimized settings
                logger.info(f"Loading model from: {model_path}")
                model = tf.keras.models.load_model(model_path, compile=False)
                model._make_predict_function()
                logger.info("Model loaded successfully")

                # prepare input tensor
                input_tensor = tf.expand_dims(processed_image, axis=0)
                if image_type == 'infrared':
                    input_tensor = tf.image.rgb_to_grayscale(input_tensor)
                    logger.info("Converting to grayscale for infrared image")

                # Make prediction with optimized settings
                logger.info("Starting prediction...")
                with tf.device('/CPU:0'):
                    prediction = model.predict(input_tensor, verbose=0)
                logger.info("Prediction completed")
                
                prediction_image = prediction[0]
                pred_arr = prediction_image[:, :, 0] if prediction_image.ndim == 3 else prediction_image
                pred_mask = (pred_arr > 0.5).astype(np.uint8)
                logger.info("Prediction mask generated")

                # Save prediction image
                result_filename = f"result_{filename}"
                result_path = os.path.join(RESULTS_FOLDER, result_filename)
                result_img = (pred_arr * 255).astype(np.uint8)
                result_pil = Image.fromarray(result_img)
                result_pil.save(result_path)

                # Convert input image to grayscale
                img_arr = processed_image.numpy()
                gray_img = img_arr.mean(axis=2)

                # True mask binarized
                if true_mask is not None:
                    ground_truth_mask = (
                        true_mask.numpy().squeeze() > 0).astype(np.uint8)
                else:
                    ground_truth_mask = (pred_arr > 0.5).astype(np.uint8)

                predicted_mask = (pred_arr > 0.5).astype(np.uint8)

                # Density map and stats
                stats_density = gray_img * predicted_mask * 100.0
                masked_density = stats_density[predicted_mask > 0]
                mean_intensity = float(
                    np.mean(masked_density)) if masked_density.size > 0 else 0.0
                std_dev = float(np.std(masked_density)
                                ) if masked_density.size > 0 else 0.0

                # Compute metrics
                intersection = int(np.logical_and(
                    ground_truth_mask, predicted_mask).sum())
                union = int(np.logical_or(
                    ground_truth_mask, predicted_mask).sum())
                iou = intersection / (union + 1e-6)
                dice = float(dice_coefficient(
                    ground_truth_mask, predicted_mask))
                precision = intersection / (predicted_mask.sum() + 1e-6)
                recall = intersection / (ground_truth_mask.sum() + 1e-6)
                spill_pixels = int(np.count_nonzero(ground_truth_mask))
                total_pixels = ground_truth_mask.size
                spill_area = float(spill_pixels / total_pixels) * 100

                metrics = {
                    'dice': dice,
                    'iou': iou,
                    'spill_pixels': spill_pixels,
                    'spill_area': spill_area,
                    'precision': float(precision),
                    'recall': float(recall),
                    'mean_intensity': mean_intensity,
                    'standard_deviation': std_dev
                }

                with open(result_path, 'rb') as img_file:
                    result_data = base64.b64encode(
                        img_file.read()).decode('utf-8')

                return jsonify({
                    'processed_image': result_data,
                    'density_graph': density_b64,
                    'metrics': metrics
                })

            except Exception as e:
                traceback.print_exc()
                return jsonify({'error': f'Error loading model or making prediction: {str(e)}'}), 500

        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=False, port=PORT)
