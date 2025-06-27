import tensorflow as tf
import os
from tensorflow.keras.utils import custom_object_scope
from custom_objects import custom_objects, CustomDiceBCELoss
import tempfile
import numpy as np

print("Starting model conversion...")
print(f"TensorFlow version: {tf.__version__}")

# Get the original model path
original_model_path = "/Users/sibikarthik/OIL_SPILL_DETECTION/1) SAR U-Net (With Augmentation)/sar_with_augmentation.keras"
output_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'sar_model_converted.tflite')

print(f"\nOriginal model path: {original_model_path}")
print(f"Output model path: {output_model_path}")

try:
    print("\n=== Loading Original Model ===")
    # Try to load the model
    try:
        # Register the custom loss function
        tf.keras.utils.register_keras_serializable()(CustomDiceBCELoss)
        
        # Load the model with custom objects
        model = tf.keras.models.load_model(
            original_model_path,
            custom_objects=custom_objects
        )
        print("Model loaded successfully as Keras model")
        
        # Print model summary
        print("\n=== Model Summary ===")
        model.summary()
        
        # Get input and output shapes
        input_shape = model.input_shape
        output_shape = model.output_shape
        print(f"\nInput shape: {input_shape}")
        print(f"Output shape: {output_shape}")
    except Exception as e:
        print(f"Error loading as Keras model: {str(e)}")
        raise
    
    # Convert to TFLite using the loaded model
    print("\n=== Converting to TFLite ===")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the model
    with open(output_model_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"\nModel converted successfully and saved to: {output_model_path}")
    
    # Test the converted model
    print("\n=== Testing Converted Model ===")
    interpreter = tf.lite.Interpreter(model_path=output_model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input details: {input_details}")
    print(f"Output details: {output_details}")
    
    # Test inference
    input_shape = input_details[0]['shape']
    input_tensor = tf.zeros(input_shape, dtype=tf.float32)
    
    interpreter.set_tensor(input_details[0]['index'], input_tensor.numpy())
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"\nInference successful")
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output min/max: {np.min(output)}, {np.max(output)}")
    
except Exception as e:
    print(f"\n=== Error ===")
    print(f"Error: {str(e)}")
    import traceback
    print("Traceback:")
    traceback.print_exc()
