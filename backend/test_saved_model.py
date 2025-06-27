import tensorflow as tf
import os

print("Testing TensorFlow SavedModel loading...")
print(f"TensorFlow version: {tf.__version__}")

# Get the current directory
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'models', 'sar_model.tflite')

print(f"\nModel path: {model_path}")
print(f"Model exists: {os.path.exists(model_path)}")

try:
    print("\n=== Loading Model ===")
    # Try to load as a saved model first
    try:
        model = tf.saved_model.load(model_path)
        print("Model loaded successfully as saved_model")
        print(f"Model signature keys: {list(model.signatures.keys())}")
    except Exception as e:
        print(f"Error loading as saved_model: {str(e)}")
        
    # Try to load as a concrete function
    try:
        print("\n=== Loading as Concrete Function ===")
        model = tf.lite.Interpreter(model_path=model_path)
        model.allocate_tensors()
        print("Model loaded successfully as TFLite interpreter")
        
        # Get input and output details
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        print(f"Input details: {input_details}")
        print(f"Output details: {output_details}")
        
        # Test inference with zeros
        input_shape = input_details[0]['shape']
        input_tensor = tf.zeros(input_shape, dtype=tf.float32)
        
        print("\n=== Testing Inference ===")
        print(f"Input tensor shape: {input_tensor.shape}")
        print(f"Input tensor dtype: {input_tensor.dtype}")
        
        # Set input and run inference
        model.set_tensor(input_details[0]['index'], input_tensor.numpy())
        model.invoke()
        output = model.get_tensor(output_details[0]['index'])
        
        print(f"\nInference successful")
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        print(f"Output min/max: {np.min(output)}, {np.max(output)}")
        
    except Exception as e:
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
