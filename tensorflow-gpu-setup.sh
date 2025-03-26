#!/bin/bash

echo "==== TensorFlow GPU Setup Script ===="
echo "This script will help fix GPU detection issues in TensorFlow"

# Check TensorFlow version
echo -e "\n1. Checking TensorFlow version..."
TF_VERSION=$(python -c "import tensorflow as tf; print(tf.__version__)")
echo "Current TensorFlow version: $TF_VERSION"

# Check CUDA version
echo -e "\n2. Checking CUDA version..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo "CUDA version: $CUDA_VERSION"
else
    echo "nvcc not found. Please make sure CUDA is properly installed."
fi

# Check for compatible TensorFlow versions
echo -e "\n3. Recommended TensorFlow versions for CUDA 12.x:"
echo "   - TensorFlow 2.15.0 or newer"
echo "   - For older CUDA versions, check https://www.tensorflow.org/install/source#gpu"

# Install compatible TensorFlow version
echo -e "\n4. Would you like to install TensorFlow 2.15.0? (y/n)"
read -p "Choice: " INSTALL_TF

if [[ $INSTALL_TF == "y" || $INSTALL_TF == "Y" ]]; then
    echo "Installing TensorFlow 2.15.0..."
    pip install tensorflow==2.15.0
fi

# Check cuDNN installation
echo -e "\n5. Checking for cuDNN installation..."
CUDNN_PATH="/usr/local/cuda-12.8/include/cudnn.h"
if [ -f "$CUDNN_PATH" ]; then
    CUDNN_VERSION=$(grep -o "CUDNN_MAJOR * \= * [0-9]" "$CUDNN_PATH" | awk '{print $3}')
    echo "cuDNN version: $CUDNN_VERSION"
else
    echo "cuDNN not found at the expected path."
    echo "Please install cuDNN from: https://developer.nvidia.com/cudnn"
fi

# Set up symbolic links if needed
echo -e "\n6. Setting up symbolic links..."
if [ ! -d "/usr/local/cuda-11.0" ]; then
    echo "Creating symbolic links from CUDA 12.8 to CUDA 11.0..."
    sudo mkdir -p /usr/local/cuda-11.0/lib64
    sudo ln -sf /usr/local/cuda-12.8/lib64/libcudart.so* /usr/local/cuda-11.0/lib64/
    sudo ln -sf /usr/local/cuda-12.8/lib64/libcublas.so* /usr/local/cuda-11.0/lib64/
    sudo ln -sf /usr/local/cuda-12.8/lib64/libcublasLt.so* /usr/local/cuda-11.0/lib64/
    sudo ln -sf /usr/local/cuda-12.8/lib64/libcufft.so* /usr/local/cuda-11.0/lib64/
    sudo ln -sf /usr/local/cuda-12.8/lib64/libcusparse.so* /usr/local/cuda-11.0/lib64/
    echo "Symbolic links created."
else
    echo "Directory /usr/local/cuda-11.0 already exists. Skipping symbolic link creation."
fi

# Update LD_LIBRARY_PATH in .bashrc
echo -e "\n7. Updating LD_LIBRARY_PATH in .bashrc..."
if ! grep -q "CUDA_HOME=/usr/local/cuda-12.8" ~/.bashrc; then
    echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc
    echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    echo "Updated .bashrc with CUDA paths."
    echo "Please run 'source ~/.bashrc' after this script completes."
else
    echo "CUDA paths already exist in .bashrc."
fi

# Install tensorflow-plugins (may help with GPU detection)
echo -e "\n8. Installing additional TensorFlow plugins..."
pip install tensorflow-io
pip install nvidia-tensorrt

echo -e "\n9. Verifying GPU detection with TensorFlow..."
python -c "import tensorflow as tf; print('GPU devices:', tf.config.list_physical_devices('GPU'))"

echo -e "\nSetup complete! If you still don't see your GPUs, try rebooting your system."
echo "You can also check the TensorFlow compatibility matrix at:"
echo "https://www.tensorflow.org/install/source#gpu"