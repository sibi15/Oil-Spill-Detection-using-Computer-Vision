import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import cloudpickle
from tensorflow.keras.utils import plot_model

# Set TensorFlow version compatibility
tf.compat.v1.disable_v2_behavior()

# Configuration
MODELS = {
    'infrared': {
        'h5_file': 'infrared_model.h5',
        'pickle_file': 'infrared_model.pkl'
    },
    'sar': {
        'h5_file': 'sar_model.h5',
        'pickle_file': 'sar_model.pkl'
    }
}

MODELS_DIR = 'models'

# Create models directory
os.makedirs(MODELS_DIR, exist_ok=True)

def convert_model_to_pickle(model_name, h5_path, pickle_path):
    """Convert HDF5 model to pickle format"""
    print(f"\nConverting {model_name}...")
    print(f"H5 model path: {h5_path}")
    
    try:
        # Load model using TensorFlow's built-in HDF5 loading
        print("Loading model...")
        model = tf.keras.models.load_model(h5_path, compile=False)
        print("Model loaded successfully")
        
        # Print model summary
        print("Model summary:")
        model.summary()
        
        # Save model as pickle
        print("Saving model as pickle...")
        with open(pickle_path, 'wb') as f:
            cloudpickle.dump({
                'model': model,
                'config': model.get_config(),
                'weights': model.get_weights(),
                'optimizer': None  # We're not saving optimizer state
            }, f)
        print("Pickle file saved")
        
        # Save model visualization
        print("Saving model architecture visualization...")
        plot_path = os.path.join(MODELS_DIR, f'{model_name}_architecture.png')
        tf.keras.utils.plot_model(model, to_file=plot_path, show_shapes=True, show_layer_names=True)
        
        print(f"Successfully converted {model_name} to {pickle_path}")
        print(f"Model architecture saved to {plot_path}")
        return True
    except Exception as e:
        print(f"Error converting {model_name}: {str(e)}")
        return False

def main():
    # Convert each model
    for model_name, config in MODELS.items():
        h5_path = os.path.join(MODELS_DIR, config['h5_file'])
        pickle_path = os.path.join(MODELS_DIR, config['pickle_file'])
        
        if os.path.exists(h5_path):
            success = convert_model_to_pickle(model_name, h5_path, pickle_path)
            if success:
                # Remove the original HDF5 model and rename the pickle file
                os.remove(h5_path)
                os.rename(pickle_path, h5_path)
                print(f"Converted and replaced {h5_path} with pickle version")
        else:
            print(f"Warning: {h5_path} not found")

if __name__ == '__main__':
    main()
