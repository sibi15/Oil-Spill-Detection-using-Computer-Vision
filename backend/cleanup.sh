#!/bin/bash

# Remove temporary test files
rm -rf test_output/

# Remove test scripts
test_scripts=(
    "test_saved_model.py"
    "test_model.py"
    "test_client.py"
    "test_app.py"
    "convert_model.py"
    "modify_model.py"
)

for script in "${test_scripts[@]}"; do
    if [ -f "$script" ]; then
        rm "$script"
        echo "Removed: $script"
    fi
done

# Remove original Keras model if it exists
if [ -f "/Users/sibikarthik/OIL_SPILL_DETECTION/1) SAR U-Net (With Augmentation)/sar_with_augmentation.keras" ]; then
    rm "/Users/sibikarthik/OIL_SPILL_DETECTION/1) SAR U-Net (With Augmentation)/sar_with_augmentation.keras"
    echo "Removed original Keras model"
fi

# Clean up pip cache
rm -rf ~/.cache/pip

# Verify cleanup
echo "Cleanup complete!"
