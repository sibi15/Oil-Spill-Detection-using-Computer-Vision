import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter
import os
import numpy as np

print("Testing TensorFlow Lite Interpreter...")
print(f"TensorFlow version: {tf.__version__}")

# Get the current directory
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'models', 'sar_model.tflite')

print(f"\nModel path: {model_path}")
print(f"Model exists: {os.path.exists(model_path)}")

try:
    print("\n=== Loading Model ===")
    # Load the model with a specific number of threads
    interpreter = Interpreter(model_path=model_path, num_threads=4)
    interpreter.allocate_tensors()
    print("Model loaded successfully")
    
    print("\n=== Model Details ===")
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"Input details: {input_details}")
    print(f"Output details: {output_details}")
    
    print("\n=== Testing Inference ===")
    # Create a dummy input tensor
    input_shape = input_details[0]['shape']
    input_tensor = np.zeros(input_shape, dtype=np.float32)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    
    # Get input and output details for verification
    print("\n=== Input Tensor Details ===")
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Input tensor dtype: {input_tensor.dtype}")
    print(f"Input tensor min/max: {np.min(input_tensor)}, {np.max(input_tensor)}")
    
    # Try inference
    try:
        print("\nAttempting inference...")
        interpreter.invoke()
        print("Inference completed")
        
        # Get output
        output = interpreter.get_tensor(output_details[0]['index'])
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        print(f"Output min/max: {np.min(output)}, {np.max(output)}")
        
    except Exception as e:
        print("\n=== Inference Error ===")
        print(f"Error during inference: {str(e)}")
        import traceback
        print("Inference traceback:")
        traceback.print_exc()
        
except Exception as e:
    print(f"\n=== Error ===")
    print(f"Error: {str(e)}")
    import traceback
    print("Traceback:")
    traceback.print_exc()
