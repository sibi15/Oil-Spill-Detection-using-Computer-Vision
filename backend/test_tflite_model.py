import tensorflow as tf
import numpy as np
import cv2
import os

def create_test_image():
    """Create a test image with a simple pattern"""
    # Create a 256x256x3 image with a simple pattern
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # Create a simple pattern (e.g., a diagonal line)
    for i in range(256):
        cv2.line(img, (0, i), (i, 0), (255, 255, 255), 1)
    
    return img

def test_model():
    # Get the current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'models', 'sar_model_converted.tflite')
    
    print(f"\n=== Loading TFLite Model ===")
    print(f"Model path: {model_path}")
    
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("\n=== Model Details ===")
    print(f"Input details: {input_details}")
    print(f"Output details: {output_details}")
    
    # Create a test image
    print("\n=== Creating Test Image ===")
    test_image = create_test_image()
    print(f"Test image shape: {test_image.shape}")
    
    # Preprocess the image
    print("\n=== Preprocessing Image ===")
    input_shape = input_details[0]['shape']
    
    # Resize if needed
    if test_image.shape[0] != input_shape[1] or test_image.shape[1] != input_shape[2]:
        test_image = cv2.resize(test_image, (input_shape[2], input_shape[1]))
    
    # Convert to float32 and normalize
    test_image = test_image.astype(np.float32) / 255.0
    
    # Add batch dimension
    test_input = np.expand_dims(test_image, axis=0)
    print(f"Processed input shape: {test_input.shape}")
    
    # Run inference
    print("\n=== Running Inference ===")
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    
    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output min/max: {np.min(output)}, {np.max(output)}")
    
    # Post-process output
    print("\n=== Post-processing Output ===")
    prediction = (output[0] > 0.5).astype(np.uint8) * 255
    print(f"Prediction shape: {prediction.shape}")
    
    # Save test images
    test_dir = os.path.join(script_dir, 'test_output')
    os.makedirs(test_dir, exist_ok=True)
    
    cv2.imwrite(os.path.join(test_dir, 'test_input.png'), test_image * 255)
    cv2.imwrite(os.path.join(test_dir, 'test_prediction.png'), prediction)
    print(f"Test images saved to: {test_dir}")

if __name__ == "__main__":
    test_model()
