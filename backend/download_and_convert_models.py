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
        'keras_file': 'infrared_model.keras',
        'pickle_file': 'infrared_model.pkl'
    },
    'sar': {
        'keras_file': 'sar_model.keras',
        'pickle_file': 'sar_model.pkl'
    }
}

MODELS_DIR = 'models'

# Create models directory
os.makedirs(MODELS_DIR, exist_ok=True)

def convert_model_to_pickle(model_name, keras_path, pickle_path):
    """Convert Keras model to pickle format"""
    print(f"\nConverting {model_name}...")
    
    # Load Keras model
    model = load_model(keras_path, compile=False)
    
    # Save model as pickle
    with open(pickle_path, 'wb') as f:
        cloudpickle.dump({
            'model': model,
            'config': model.get_config(),
            'weights': model.get_weights(),
            'optimizer': model.optimizer.get_config() if model.optimizer else None
        }, f)
    
    # Save model visualization
    plot_path = os.path.join(MODELS_DIR, f'{model_name}_architecture.png')
    plot_model(model, to_file=plot_path, show_shapes=True, show_layer_names=True)
    
    print(f"Successfully converted {model_name} to {pickle_path}")
    print(f"Model architecture saved to {plot_path}")

def main():
    # Download and convert each model
    for model_name, config in MODELS.items():
        keras_path = os.path.join(MODELS_DIR, config['keras_file'])
        pickle_path = os.path.join(MODELS_DIR, config['pickle_file'])
        
        # Download Keras model if it doesn't exist
        if not os.path.exists(keras_path):
            download_file(config['url'], keras_path)
        
        # Convert to pickle if it doesn't exist
        if not os.path.exists(pickle_path):
            convert_model_to_pickle(model_name, keras_path, pickle_path)
        else:
            print(f"\n{pickle_path} already exists, skipping conversion")
            
        # Clean up Keras model file
        if os.path.exists(keras_path):
            os.remove(keras_path)
            print(f"Removed {keras_path}")

if __name__ == '__main__':
    main()
