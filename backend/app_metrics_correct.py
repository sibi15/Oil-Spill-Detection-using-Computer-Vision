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

app = Flask(__name__)
CORS(app)

# Set folders
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
MODEL_FOLDER = 'models'
LABELS_FOLDER = 'labels'
OUTPUT_SIZE = (512, 512)
SAMPLE_SIZE = (256, 256)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(LABELS_FOLDER, exist_ok=True)

matplotlib.use('Agg')


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
            # select model path; infrared uses explicit path
            if image_type == 'infrared':
                model_path = r"C:\Users\ADMIN\Documents\sibi\new\project\backend\models\infrared_model.keras"
            else:
                model_path = os.path.join(
                    MODEL_FOLDER, f"{image_type}_model.keras")

            if not os.path.exists(model_path):
                return jsonify({'error': f'Model for {image_type} not found. Please add the model to {model_path}'}), 404

            try:
                model = tf.keras.models.load_model(model_path, compile=False)
                # prepare input tensor
                input_tensor = tf.expand_dims(processed_image, axis=0)
                if image_type == 'infrared':
                    # convert RGB to single-channel grayscale for infrared model
                    input_tensor = tf.image.rgb_to_grayscale(input_tensor)
                prediction = model.predict(input_tensor)
                # raw predicted mask (0–1 values) from UNet model
                pred_mask = prediction[0, :, :, 0]
                prediction_image = prediction[0]

                result_filename = f"result_{filename}"
                result_path = os.path.join(RESULTS_FOLDER, result_filename)

                if len(prediction_image.shape) == 3 and prediction_image.shape[-1] == 1:
                    result_img = (
                        prediction_image[:, :, 0] * 255).astype(np.uint8)
                    result_pil = Image.fromarray(result_img)
                else:
                    result_img = (prediction_image * 255).astype(np.uint8)
                    result_pil = Image.fromarray(result_img)

                result_pil.save(result_path)

                pred_arr = (
                    prediction_image[:, :, 0] if prediction_image.ndim == 3 else prediction_image)
                pred_mask = (pred_arr > 0.5).astype(np.uint8)
                binary_mask = (true_mask.numpy() > 0).astype(
                    np.uint8) if true_mask is not None else pred_mask

                # plot and encode density graph using processed_image and true_mask
                img_arr = processed_image.numpy()
                # grayscale representation
                gray_img = img_arr.mean(axis=2)
                # binary ground-truth mask
                mask_arr = true_mask.numpy().squeeze() if true_mask is not None else pred_mask
                # density map percent (0–100)
                density_map = gray_img * mask_arr * 100.0
                normed_uint8 = density_map.astype(np.uint8)
                # create figure
                plt.figure(figsize=(12, 3))
                plt.subplot(1, 4, 1)
                plt.title('Original Oil Spill')
                plt.imshow(gray_img, cmap='gray')
                plt.axis('off')
                plt.subplot(1, 4, 2)
                plt.title('True Spill Mask')
                plt.imshow(mask_arr, cmap='gray')
                plt.axis('off')
                plt.subplot(1, 4, 3)
                plt.title('Predicted Binary Spill Mask')
                plt.imshow(pred_mask, cmap='gray')
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

                # compute intensity stats on raw density
                mean_intensity = float(np.mean(density_map))
                std_dev = float(np.std(density_map))

                # preserve ground truth mask
                gt_mask = binary_mask.copy()

                # generate predicted binary mask via Otsu thresholding
                pred_mask_8bit = (pred_mask * 255).astype(np.uint8)
                _, pred_binary = cv2.threshold(
                    pred_mask_8bit, 0, 255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                pred_binary = cv2.resize(
                    pred_binary,
                    (gt_mask.shape[1], gt_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST)
                pred_binary = (pred_binary / 255).astype(np.uint8)

                # compute intersection and union
                intersection = int(np.logical_and(gt_mask, pred_binary).sum())
                union = int(np.logical_or(gt_mask, pred_binary).sum())
                iou = intersection / (union + 1e-6)

                # compute dice, precision, recall
                dice = (2. * intersection) / \
                    (np.sum(gt_mask) + np.sum(pred_binary) + 1e-6)
                precision = intersection / (np.sum(pred_binary) + 1e-6)
                recall = intersection / (np.sum(gt_mask) + 1e-6)

                metrics = {
                    'dice': float(dice),
                    'iou': float(iou),
                    'spill_area': int(np.sum(gt_mask)),
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
