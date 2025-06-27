import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import psutil
import requests
import time
from skimage.transform import resize
from PIL import Image
import tensorflow as tf
from tensorflow.lite import Interpreter
from werkzeug.utils import secure_filename
import traceback  # for debug exception tracing
import requests
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', '/tmp/uploads')
RESULTS_FOLDER = os.getenv('RESULTS_FOLDER', '/tmp/results')
MODEL_FOLDER = os.getenv('MODEL_FOLDER', 'models')
LABELS_FOLDER = os.getenv('LABELS_FOLDER', '/tmp/labels')
OUTPUT_SIZE = (512, 512)
SAMPLE_SIZE = (256, 256)

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(LABELS_FOLDER, exist_ok=True)

# Model configuration
# For local development
LOCAL_MODEL_PATH = os.path.join(MODEL_FOLDER, 'sar_model.keras')

# For deployment on Render - model will be downloaded from GitHub Releases
# Using TensorFlow Lite# Use local model directly from repository
MODEL_DOWNLOAD_URL = None  # No need to download
# Use a more reliable path for the model
# The model should be in the project root directory
DEPLOY_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'sar_model.tflite')
print(f"Model path: {DEPLOY_MODEL_PATH}")
print(f"Model exists: {os.path.exists(DEPLOY_MODEL_PATH)}")
print(f"Current directory: {os.getcwd()}")
print(f"Directory contents: {os.listdir(os.path.dirname(DEPLOY_MODEL_PATH))}")

# Verify model file exists and is readable
if not os.path.exists(DEPLOY_MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {DEPLOY_MODEL_PATH}")
if not os.access(DEPLOY_MODEL_PATH, os.R_OK):
    raise PermissionError(f"Cannot read model file at {DEPLOY_MODEL_PATH}")

# Log model file details
print(f"Using model file: {DEPLOY_MODEL_PATH}")
print(f"Model file size: {os.path.getsize(DEPLOY_MODEL_PATH)} bytes")

# Create models directory if it doesn't exist
os.makedirs(os.path.dirname(DEPLOY_MODEL_PATH), exist_ok=True)

def download_model():
    """Check if local model exists and verify it's readable"""
    if os.path.exists(DEPLOY_MODEL_PATH):
        try:
            # Try to open and read a small part of the file
            with open(DEPLOY_MODEL_PATH, 'rb') as f:
                f.read(1024)  # Read first 1KB
            logger.info("Using local model")
            return True
        except Exception as e:
            logger.error(f"Error accessing model file: {str(e)}")
            return False
    logger.error("Local model not found")
    return False

# Model loading utilities
def get_model():
    """Get or create the TensorFlow Lite model instance with memory optimization"""
    global interpreter, input_details, output_details
    
    if interpreter is None:
        try:
            # Verify model file exists and is readable
            if not os.path.exists(DEPLOY_MODEL_PATH):
                raise FileNotFoundError(f"Model file not found at {DEPLOY_MODEL_PATH}")
                
            # Load model with minimal memory usage
            interpreter = Interpreter(model_path=DEPLOY_MODEL_PATH, num_threads=1)
            interpreter.allocate_tensors()
            
            # Get model details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Log memory usage
            current_memory = psutil.virtual_memory().used / (1024 * 1024)
            logger.info(f"Model loaded successfully. Memory usage: {current_memory:.1f} MB")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise e
    
    return interpreter, input_details, output_details

# Initialize model as None - will be loaded on first request
interpreter = None
input_details = None
output_details = None

# Server configuration
PORT = 8080
app.config['ENV'] = 'production'
app.config['DEBUG'] = False

# Ensure directories exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('labels', exist_ok=True)

# Initialize Flask app with proper port binding
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)

# Ensure directories exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('labels', exist_ok=True)

# Initialize Flask app with proper port binding
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)
# Remove model download function since we're using local model
# Model should be included in git repository
PYTHONUNBUFFERED = os.getenv('PYTHONUNBUFFERED', '1')  # Enable unbuffered output for better logging

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
        print("\n=== Received Headers ===")
        print(request.headers)

        print("\n=== Received Files ===")
        print(request.files)

        print("\n=== Form Data ===")
        print(request.form)

        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        image_type = request.form.get('imageType')

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # read raw grayscale image for plotting and density
            original_image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            # load processed image and optional true mask
            processed_image = load_images(filepath)
            true_mask = None
            original_label = None

            # attempt auto-load static mask
            base = os.path.splitext(filename)[0]
            lbl_static = os.path.join(LABELS_FOLDER, f"{base}.png")
            if os.path.exists(lbl_static):
                original_label = cv2.imread(lbl_static, cv2.IMREAD_GRAYSCALE)
                processed_image, true_mask = load_images(filepath, lbl_static)

            # override with uploaded label if provided
            label_file = request.files.get('label')
            if label_file:
                os.makedirs(LABELS_FOLDER, exist_ok=True)
                lbl_fname = secure_filename(label_file.filename)
                lbl_path = os.path.join(LABELS_FOLDER, lbl_fname)
                label_file.save(lbl_path)
                original_label = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
                processed_image, true_mask = load_images(filepath, lbl_path)

            # Get model instance
            interpreter, input_details, output_details = get_model()
            
            # Prepare input tensor
            input_tensor = tf.expand_dims(processed_image, axis=0)
            
            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], input_tensor.numpy())
            
            # Run inference
            interpreter.invoke()
            
            # Get output tensor
            output = interpreter.get_tensor(output_details[0]['index'])
            
            prediction_image = output[0]
            pred_arr = prediction_image[:, :, 0] if prediction_image.ndim == 3 else prediction_image
            pred_mask = (pred_arr > 0.5).astype(np.uint8)

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
            std_dev = float(np.std(masked_density)) if masked_density.size > 0 else 0.0

            # Create visualization
            normed_uint8 = stats_density.astype(np.uint8)
            plt.figure(figsize=(12, 3))
            plt.subplot(1, 4, 1)
            plt.title('Original Oil Spill')
            plt.imshow(gray_img, cmap='gray')
            plt.axis('off')
            plt.subplot(1, 4, 2)
            plt.title('True Spill Mask')
            plt.imshow(ground_truth_mask, cmap='gray')
            plt.axis('off')
            plt.subplot(1, 4, 3)
            plt.title('Predicted Binary Spill Mask')
            plt.imshow(predicted_mask, cmap='gray')
            plt.axis('off')
            plt.subplot(1, 4, 4)
            plt.title('Density Map (0–100)')
            img = plt.imshow(normed_uint8, cmap='jet', vmin=0, vmax=100)
            plt.colorbar(img, ticks=[0, 20, 40, 60, 80, 100])
            plt.axis('off')
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            density_b64 = base64.b64encode(buf.read()).decode('utf-8')

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
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=False, port=PORT, host='0.0.0.0', use_reloader=False)
