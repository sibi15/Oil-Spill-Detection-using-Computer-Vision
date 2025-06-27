import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter
import os

# Get the current directory
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'models', 'sar_model.tflite')

print(f"Testing model at: {model_path}")
print(f"Model exists: {os.path.exists(model_path)}")

try:
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    print("\n=== Model Details ===")
    print(f"Input details: {interpreter.get_input_details()}")
    print(f"Output details: {interpreter.get_output_details()}")
    print("\nModel loaded successfully!")
except Exception as e:
    print(f"\n=== Error Loading Model ===")
    print(f"Error: {str(e)}")
    import traceback
    print("Traceback:")
    traceback.print_exc()
