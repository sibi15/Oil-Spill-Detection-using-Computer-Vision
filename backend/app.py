import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import numpy as np
import psutil
import matplotlib
matplotlib.use('Agg')  # headless server: no GUI backend
import matplotlib.pyplot as plt
import io
import base64
import threading
from PIL import Image
from werkzeug.exceptions import HTTPException

# Only the TFLite interpreter is needed at inference time. The standalone LiteRT
# runtime is a few MB, where full TensorFlow costs hundreds of MB per worker and
# runs a small (512 MB) instance out of memory.
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": "*", "expose_headers": "*"}}, supports_credentials=True)

# Configuration
OUTPUT_SIZE = (512, 512)
SAMPLE_SIZE = (256, 256)
PORT = 8080
MAX_UPLOAD_MB = 20
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024


# Model configuration
DEPLOY_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'sar_model_converted.tflite')

# Verify model file exists and is readable
if not os.path.exists(DEPLOY_MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {DEPLOY_MODEL_PATH}")
if not os.access(DEPLOY_MODEL_PATH, os.R_OK):
    raise PermissionError(f"Cannot read model file at {DEPLOY_MODEL_PATH}")

# Initialize model variables
global interpreter, input_details, output_details
interpreter = None
input_details = None
output_details = None

# Model loading utilities
def get_model():
    # Get or create the TensorFlow Lite model instance with memory optimization
    global interpreter, input_details, output_details
    
    if interpreter is None:
        try:
            logger.info(f"Loading model from {DEPLOY_MODEL_PATH}")
            interpreter = Interpreter(model_path=DEPLOY_MODEL_PATH, num_threads=1)
            interpreter.allocate_tensors()
            logger.info("Model tensors allocated")
            
            # Get model details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            logger.info(f"Model input details: {input_details}")
            logger.info(f"Model output details: {output_details}")
            
            # Log this worker's memory use
            current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            logger.info(f"Model loaded successfully. Worker memory: {current_memory:.1f} MB")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise e
    
    return interpreter, input_details, output_details

# Environment variable for unbuffered output
PYTHONUNBUFFERED = os.getenv('PYTHONUNBUFFERED', '1')

def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-6)


def classify_density(x):
    return x  # simple pass-through density classifier


def decode_image(data):
    """Decode an upload to an RGB uint8 array without holding a full-size photo in memory.

    A 12 MP phone photo decodes to ~36 MB per copy, enough to push a small instance
    over its memory limit. The model only needs 256 x 256, so JPEGs are decoded at
    1/2, 1/4 or 1/8 scale (libjpeg does this cheaply) while staying above 512 px.
    """
    flag = cv2.IMREAD_COLOR
    try:
        w, h = Image.open(io.BytesIO(data)).size  # reads the header only
        short_side = min(w, h)
        for scale, reduced in ((8, cv2.IMREAD_REDUCED_COLOR_8),
                               (4, cv2.IMREAD_REDUCED_COLOR_4),
                               (2, cv2.IMREAD_REDUCED_COLOR_2)):
            if short_side // scale >= 512:
                flag = reduced
                break
    except Exception:
        pass  # unknown header: let OpenCV decide whether it can read it
    image = cv2.imdecode(np.frombuffer(data, np.uint8), flag)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@app.errorhandler(413)
def too_large(_):
    return jsonify({'error': f'Image is too large (max {MAX_UPLOAD_MB} MB)'}), 413


@app.route('/')
def health_check():
    logger.info("Health check request received")
    return jsonify({"status": "healthy", "version": "1.0.0"})

# One interpreter per process, shared by request threads. Neither the TFLite
# interpreter nor matplotlib's pyplot is thread-safe, and the model's working
# memory (~300 MB) only fits a small instance once, so predictions run one at a time.
PREDICT_LOCK = threading.Lock()


@app.route('/predict', methods=['POST'])
def predict():
    with PREDICT_LOCK:
        return _predict()


def _predict():
    try:
        logger.info("New prediction request received")
        logger.info(f"Files: {request.files}")
        logger.info(f"Form: {request.form}")
        
        # Check if file was uploaded
        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.error("Empty file name")
            return jsonify({'error': 'No selected file'}), 400
        
        # Get image type
        image_type = request.form.get('imageType', 'sar')
        logger.info(f"Processing image type: {image_type}")
        
        # Decode in memory: nothing is written to disk
        try:
            image = decode_image(file.read())
            if image is None:
                logger.error("Failed to decode upload")
                return jsonify({'error': 'Failed to read image'}), 400
            
            # Resize to model input size
            processed_image = cv2.resize(image, (256, 256))
            processed_image = processed_image.astype(np.float32) / 255.0
            logger.info(f"Image processed successfully. Shape: {processed_image.shape}")
            
            # Get model instance
            interpreter, input_details, output_details = get_model()
            
            # Prepare input tensor
            input_tensor = np.expand_dims(processed_image, axis=0)
            
            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            logger.info(f"Input tensor set successfully. Shape: {input_tensor.shape}")
            
            # Run inference
            try:
                logger.info("Starting model inference...")
                interpreter.invoke()
                logger.info("Inference completed successfully")
            except Exception as e:
                logger.error(f"Error during inference: {str(e)}")
                raise e
            
            # Get output tensor
            output = interpreter.get_tensor(output_details[0]['index'])
            logger.info(f"Output tensor shape: {output.shape}")
            
            # Process output
            # Output is (1, 256, 256, 1): drop the batch and channel axes to get a 2-D mask.
            # Keeping the channel axis would broadcast against the 2-D image below
            # into a (256, 256, 256) array.
            prediction = (output[0, :, :, 0] > 0.5).astype(np.uint8)
            logger.info(f"Prediction processed successfully. Shape: {prediction.shape}")
            
            # Create visualization
            img_arr = processed_image
            gray_img = img_arr.mean(axis=2)
            ground_truth_mask = prediction.copy()
            predicted_mask = prediction.copy()
            
            # Density map and stats
            stats_density = gray_img * predicted_mask * 100.0
            masked_density = stats_density[predicted_mask > 0]
            mean_intensity = float(np.mean(masked_density)) if masked_density.size > 0 else 0.0
            std_dev = float(np.std(masked_density)) if masked_density.size > 0 else 0.0
            
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
            processed_image = base64.b64encode(buf.read()).decode('utf-8')
            
            # Create density graph separately
            density_map = stats_density.astype(np.uint8)
            plt.figure(figsize=(12, 3))
            plt.imshow(density_map, cmap='jet', vmin=0, vmax=100)
            plt.colorbar(ticks=[0, 20, 40, 60, 80, 100])
            plt.title('Density Map (0–100)')
            plt.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            density_graph = base64.b64encode(buf.read()).decode('utf-8')

            # Compute metrics
            intersection = int(np.logical_and(ground_truth_mask, predicted_mask).sum())
            union = int(np.logical_or(ground_truth_mask, predicted_mask).sum())
            iou = intersection / (union + 1e-6)
            dice = (2. * intersection) / (np.sum(ground_truth_mask) + np.sum(predicted_mask) + 1e-6)
            precision = intersection / (predicted_mask.sum() + 1e-6)
            recall = intersection / (ground_truth_mask.sum() + 1e-6)
            spill_pixels = int(np.count_nonzero(ground_truth_mask))
            total_pixels = ground_truth_mask.size
            spill_area = float(spill_pixels / total_pixels) * 100

            metrics = {
                'dice': float(dice),
                'iou': float(iou),
                'spill_pixels': spill_pixels,
                'spill_area': spill_area,
                'precision': float(precision),
                'recall': float(recall),
                'mean_intensity': mean_intensity,
                'standard_deviation': std_dev
            }
            logger.info(f"Metrics calculated: {metrics}")
            
            return jsonify({
                'processed_image': processed_image,
                'density_graph': density_graph,
                'metrics': metrics
            })
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise e
            
    except HTTPException:
        raise  # e.g. 413 for an oversized upload: keep its status and message
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        import traceback
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Error processing image'}), 500

if __name__ == '__main__':
    try:
        # Get port from environment variable (Render uses port 8080 by default)
        port = int(os.environ.get('PORT', 8080))
        
        print(f"\nStarting Flask app on port {port}")
        print(f"Model path: {DEPLOY_MODEL_PATH}")
        
        # Initialize model
        get_model()
        
        # Run the app
        app.run(
            host='0.0.0.0',
            port=port,
            debug=False,
            use_reloader=False
        )
    except Exception as e:
        print(f"\n=== App Startup Error ===")
        print(f"Error: {str(e)}")
        import traceback
        print("Startup traceback:")
        traceback.print_exc()
