import tensorflow as tf
import os
import logging
import numpy as np
import h5py
from tensorflow.keras import layers, models
from tensorflow.keras.utils import CustomObjectScope

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the correct paths
HDF5_PATH = '/Users/sibikarthik/Downloads/Oil-Spill-Detection-using-Computer-Vision/backend/models/sar_model.h5'
SAVEDMODEL_PATH = '/Users/sibikarthik/Downloads/Oil-Spill-Detection-using-Computer-Vision/backend/models/saved_model'
TFLITE_PATH = '/Users/sibikarthik/Downloads/Oil-Spill-Detection-using-Computer-Vision/backend/models/sar_model.tflite'

# Define custom objects for the model
custom_objects = {
    'dice_bce_mc_loss': lambda y_true, y_pred: tf.reduce_mean(
        y_true * -tf.math.log(y_pred + 1e-7) + 
        (1 - y_true) * -tf.math.log(1 - y_pred + 1e-7)
    ),
    'dice_mc_metric': lambda y_true, y_pred: tf.reduce_mean(
        (2 * tf.reduce_sum(y_true * y_pred, axis=[1,2,3]) + 1e-7) / 
        (tf.reduce_sum(y_true, axis=[1,2,3]) + tf.reduce_sum(y_pred, axis=[1,2,3]) + 1e-7)
    )
}

def create_compatible_model(input_shape):
    """Create a simple compatible model architecture"""
    inputs = layers.Input(shape=input_shape)
    
    # Simple encoder
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    
    # Simple decoder
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.UpSampling2D()(x)
    
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.UpSampling2D()(x)
    
    # Output layer
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Decoder
    x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2DTranspose(128, 4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    outputs = layers.Conv2DTranspose(1, 4, strides=2, padding='same', activation='sigmoid')(x)
    
    return models.Model(inputs=inputs, outputs=outputs)

def load_weights_from_h5(model, h5_path):
    """Load weights from HDF5 file directly"""
    try:
        with h5py.File(h5_path, 'r') as f:
            # Get input shape from model config
            model_config = f.attrs['model_config']
            if isinstance(model_config, bytes):
                model_config = model_config.decode('utf-8')
            
            # Load weights
            logger.info("Loading model weights...")
            for layer in model.layers:
                if layer.name in f['model_weights']:
                    layer_weights = []
                    for weight in f['model_weights'][layer.name][layer.name]:
                        layer_weights.append(np.array(f['model_weights'][layer.name][layer.name][weight]))
                    layer.set_weights(layer_weights)
            logger.info("Weights loaded successfully")
            
            return True
    except Exception as e:
        logger.error(f"Error loading weights: {str(e)}")
        return False

def convert_model():
    """Convert HDF5 model to TensorFlow Lite format"""
    try:
        # Create new compatible model architecture
        logger.info("Creating compatible model architecture")
        compatible_model = create_compatible_model((256, 256, 3))
        logger.info("Compatible model created")

        # Load weights from HDF5 file
        logger.info(f"Loading weights from {HDF5_PATH}")
        if not load_weights_from_h5(compatible_model, HDF5_PATH):
            raise Exception("Failed to load weights")
        logger.info("Weights loaded successfully")

        # Convert to SavedModel format
        logger.info(f"Converting to SavedModel format at {SAVEDMODEL_PATH}")
        tf.saved_model.save(compatible_model, SAVEDMODEL_PATH)
        logger.info("SavedModel conversion completed successfully!")

        # Convert to TensorFlow Lite
        logger.info("Converting to TensorFlow Lite format...")
        converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_PATH)
        
        # Set optimization options
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS    # Enable TensorFlow ops.
        ]
        
        # Convert model
        tflite_model = converter.convert()
        
        # Save the model
        logger.info(f"Saving TensorFlow Lite model to {TFLITE_PATH}")
        with open(TFLITE_PATH, 'wb') as f:
            f.write(tflite_model)
        
        logger.info("Conversion completed successfully!")
        logger.info(f"Original HDF5 model size: {os.path.getsize(HDF5_PATH)/1024/1024:.1f} MB")
        logger.info(f"TensorFlow Lite model size: {os.path.getsize(TFLITE_PATH)/1024/1024:.1f} MB")
        
        return True

    except Exception as e:
        logger.error(f"Error during conversion: {str(e)}")
        return False
        return False

if __name__ == "__main__":
    convert_model()
    raise