import os


def setup_cuda():
    # Set environment variables before importing any other libraries
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

    # Add CUDA to the path
    os.environ["PATH"] = "/usr/local/cuda-12.8/bin:" + os.environ.get("PATH", "")

    # Important: Add CUDA's nvcc to the system path
    os.environ["CUDA_PATH"] = "/usr/local/cuda-12.8"

    # Enable TensorFlow to see the GPU
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"