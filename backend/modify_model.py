import tensorflow as tf
import os
from tensorflow.keras.utils import custom_object_scope
from custom_objects import custom_objects

print("Starting model modification...")
print(f"TensorFlow version: {tf.__version__}")

# Get the current directory
script_dir = os.path.dirname(os.path.abspath(__file__))
original_model_path = os.path.join(script_dir, 'models', 'sar_model.h5')
modified_model_path = os.path.join(script_dir, 'models', 'sar_model_modified.h5')

print(f"\nOriginal model path: {original_model_path}")
print(f"Modified model path: {modified_model_path}")

try:
    print("\n=== Loading Original Model ===")
    # Load the model with custom objects
    with custom_object_scope(custom_objects):
        model = tf.keras.models.load_model(original_model_path)
        print("Model loaded successfully as Keras model")
        
        # Print model summary
        print("\n=== Original Model Summary ===")
        model.summary()
        
        # Get input and output shapes
        input_shape = model.input_shape
        output_shape = model.output_shape
        print(f"\nInput shape: {input_shape}")
        print(f"Output shape: {output_shape}")
        
        # Create a new model with the same architecture but without custom loss
        print("\n=== Creating Modified Model ===")
        modified_model = tf.keras.models.clone_model(model)
        modified_model.build(input_shape)
        
        # Compile with standard loss function
        modified_model.compile(
            optimizer=model.optimizer,
            loss='binary_crossentropy',  # Using standard loss instead of custom
            metrics=['accuracy']
        )
        
        print("\n=== Modified Model Summary ===")
        modified_model.summary()
        
        # Save the modified model
        modified_model.save(modified_model_path)
        print(f"\nModified model saved to: {modified_model_path}")
        
except Exception as e:
    print(f"\n=== Error ===")
    print(f"Error: {str(e)}")
    import traceback
    print("Traceback:")
    traceback.print_exc()
