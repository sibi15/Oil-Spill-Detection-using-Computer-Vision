import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = '/tmp/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ENV'] = 'development'
app.config['DEBUG'] = True
app.config['CORS_HEADERS'] = 'Content-Type'
DEPLOY_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'sar_model.tflite')

# Ensure directories exist
os.makedirs('uploads', exist_ok=True)

# Server configuration
PORT = 5001

# Model loading utilities
def get_model():
    """Get or create the TensorFlow Lite model instance with memory optimization"""
    global interpreter, input_details, output_details
    
    if interpreter is None:
        try:
            logger.info(f"Loading model from {DEPLOY_MODEL_PATH}")
            if not os.path.exists(DEPLOY_MODEL_PATH):
                logger.error(f"Model file not found at {DEPLOY_MODEL_PATH}")
                raise FileNotFoundError(f"Model file not found at {DEPLOY_MODEL_PATH}")
                
            logger.info("Loading TensorFlow Lite model...")
            interpreter = Interpreter(model_path=DEPLOY_MODEL_PATH, num_threads=1)
            interpreter.allocate_tensors()
            logger.info("Model tensors allocated")
            
            # Get model details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            logger.info(f"Model input details: {input_details}")
            logger.info(f"Model output details: {output_details}")
            
            # Log memory usage
            current_memory = psutil.virtual_memory().used / (1024 * 1024)
            logger.info(f"Model loaded successfully. Memory usage: {current_memory:.1f} MB")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise e
    
    return interpreter, input_details, output_details
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter
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
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": "*", "expose_headers": "*"}}, supports_credentials=True)

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
PORT = 5000
app.config['ENV'] = 'development'
app.config['DEBUG'] = True
app.config['CORS_HEADERS'] = 'Content-Type'

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


@app.route('/')
def health_check():
    logger.info("Health check request received")
    return jsonify({"status": "healthy", "version": "1.0.0"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("New prediction request received")
        logger.info(f"Headers: {dict(request.headers)}")
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
        
        # Save uploaded file temporarily
        temp_path = os.path.join('uploads', file.filename)
        file.save(temp_path)
        logger.info(f"File saved to: {temp_path}")
        
        # Load and preprocess image
        try:
            image = cv2.imread(temp_path)
            if image is None:
                logger.error(f"Failed to read image from {temp_path}")
                return jsonify({'error': 'Failed to read image'}), 400
            
            # Convert to RGB if needed
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size
            processed_image = cv2.resize(image, (256, 256))
            processed_image = processed_image.astype(np.float32) / 255.0
            logger.info(f"Image processed successfully. Shape: {processed_image.shape}")
            
            # Get model instance
            interpreter, input_details, output_details = get_model()
            
            # Verify input shape
            input_shape = input_details[0]['shape']
            logger.info(f"\n=== Model Input Shape ===")
            logger.info(f"Expected shape: {input_shape}")
            logger.info(f"Actual shape: {processed_image.shape}")
            
            # Prepare input tensor with correct shape
            target_height = input_shape[1]
            target_width = input_shape[2]
            
            if processed_image.shape[0] != target_height or processed_image.shape[1] != target_width:
                logger.info(f"Resizing input to match model expected shape ({target_height}, {target_width})")
                processed_image = tf.image.resize(processed_image, (target_height, target_width))
            
            # Ensure correct data type
            processed_image = tf.cast(processed_image, tf.float32)
            
            input_tensor = tf.expand_dims(processed_image, axis=0)
            
            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], input_tensor.numpy())
            logger.info(f"Input tensor set successfully. Shape: {input_tensor.shape}")
            
            # Run inference
            try:
                logger.info("Starting model inference...")
                interpreter.invoke()
                logger.info("Inference completed successfully")
            except Exception as e:
                logger.error(f"\n=== Inference Error Details ===")
                logger.error(f"Input tensor shape: {input_tensor.shape}")
                logger.error(f"Input details: {input_details[0]}")
                logger.error(f"Input tensor dtype: {input_tensor.dtype}")
                logger.error(f"Input tensor min/max: {tf.reduce_min(input_tensor)}, {tf.reduce_max(input_tensor)}")
                raise e
            
            # Get output tensor
            output = interpreter.get_tensor(output_details[0]['index'])
            logger.info(f"Output tensor shape: {output.shape}")
            
            # Process output
            prediction = (output[0] > 0.5).astype(np.uint8)
            logger.info(f"Prediction processed successfully. Shape: {prediction.shape}")
            
            # Calculate metrics
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
                'mean': float(np.mean(prediction)),
                'std': float(np.std(prediction)),
                'max': float(np.max(prediction)),
                'min': float(np.min(prediction))
            }
            logger.info(f"Metrics calculated: {metrics}")
            
            # Clean up
            os.remove(temp_path)
            logger.info(f"Temporary file removed: {temp_path}")
            
            return jsonify({
                'success': True,
                'metrics': metrics,
                'timestamp': time.time()
            })
        
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise
    
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

if __name__ == '__main__':
    try:
        print(f"\nStarting Flask app on port {PORT}")
        print(f"Model path: {DEPLOY_MODEL_PATH}")
        print(f"Model exists: {os.path.exists(DEPLOY_MODEL_PATH)}")
        print(f"Current directory: {os.getcwd()}")
        print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
        
        # Test if we can load the model
        try:
            print("\n=== Testing Model Loading ===")
            interpreter = Interpreter(model_path=DEPLOY_MODEL_PATH, num_threads=1)
            interpreter.allocate_tensors()
            print("Model loaded successfully")
            print(f"Input details: {interpreter.get_input_details()}")
            print(f"Output details: {interpreter.get_output_details()}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            import traceback
            print("Model loading traceback:")
            traceback.print_exc()
            raise
        
        print("\n=== Starting Flask App ===")
        app.run(debug=True, port=PORT, host='0.0.0.0', use_reloader=False)
    except Exception as e:
        print(f"\n=== App Startup Error ===")
        print(f"Error: {str(e)}")
        import traceback
        print("Startup traceback:")
        traceback.print_exc()
