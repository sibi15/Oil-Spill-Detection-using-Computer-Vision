from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib
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
# from flask_wtf.csrf import CSRFProtect
import traceback  # for debug exception tracing
import requests

app = Flask(__name__)
CORS(app)

# Set folders
UPLOAD_FOLDER = '/tmp/uploads'
RESULTS_FOLDER = '/tmp/results'
MODEL_FOLDER = 'models'
LABELS_FOLDER = '/tmp/labels'
OUTPUT_SIZE = (512, 512)
SAMPLE_SIZE = (256, 256)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(LABELS_FOLDER, exist_ok=True)

matplotlib.use('Agg')

# Model URLs - you'll need to upload your models to a cloud storage service
MODEL_URLS = {
    'infrared_model.keras': 'https://drive.google.com/uc?export=download&id=1azpgoH2M52HQtjNj3V_aeLzy_X5xgsXu',
    'sar_model.keras': 'https://drive.google.com/uc?export=download&id=1le5uHObuGbiQKyw_r8p9JgY_6eko-9h1'
}

def download_model_if_needed(model_name: str):
    """Download model if it doesn't exist locally."""
    model_path = os.path.join(MODEL_FOLDER, model_name)
    
    if os.path.exists(model_path):
        return model_path
    
    url = MODEL_URLS.get(model_name)
    if not url or url == f'YOUR_{model_name.upper().replace(".", "_")}_URL_HERE':
        return None
    
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

        try:
            # select model path; use cross-platform path
            if image_type == 'infrared':
                model_path = download_model_if_needed("infrared_model.keras")
            else:
                model_path = download_model_if_needed(f"{image_type}_model.keras")

            if not model_path:
                return jsonify({'error': f'Model for {image_type} not found. Please add the model to {model_path}'}), 404

            try:
                model = tf.keras.models.load_model(model_path, compile=False)

                # prepare input tensor
                input_tensor = tf.expand_dims(processed_image, axis=0)
                if image_type == 'infrared':
                    input_tensor = tf.image.rgb_to_grayscale(input_tensor)

                prediction = model.predict(input_tensor)
                prediction_image = prediction[0]
                pred_arr = prediction_image[:, :,
                                            0] if prediction_image.ndim == 3 else prediction_image
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
                std_dev = float(np.std(masked_density)
                                ) if masked_density.size > 0 else 0.0

                # Plot visualizations
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

                # include density graph image
                density_key = density_b64 if 'density_b64' in locals() else ''
                return jsonify({
                    'processed_image': result_data,
                    'density_graph': density_key,
                    'metrics': metrics
                })

            except Exception as e:
                traceback.print_exc()
                return jsonify({'error': f'Error loading model or making prediction: {str(e)}'}), 500

        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': f'Error preprocessing image: {str(e)}'}), 500


if __name__ == '__main__':
    # app.run(debug=True)
    app.run(debug=True, port=5000)

# For Vercel deployment
app.debug = False
