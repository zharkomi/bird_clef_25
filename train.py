# Enable faulthandler for better debugging
import faulthandler
import os
import random
import traceback

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

faulthandler.enable()
import logging

# Now import TensorFlow and check if it can see GPUs
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Try to force TensorFlow to detect the GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        # Enable memory growth to avoid allocating all GPU memory at once
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Memory growth enabled for {len(physical_devices)} GPUs")
    except Exception as e:
        print(f"Error enabling memory growth: {e}")
else:
    print("\nNo GPUs found. Checking system configuration...")
    # Print diagnostic information
    try:
        import subprocess

        print("\nChecking NVIDIA driver with nvidia-smi:")
        nvidia_smi = subprocess.run(["nvidia-smi"], text=True, capture_output=True)
        print(nvidia_smi.stdout)

        print("\nChecking CUDA installation:")
        nvcc = subprocess.run(["nvcc", "--version"], text=True, capture_output=True)
        print(nvcc.stdout)

        print("\nChecking TensorFlow build information:")
        print(tf.sysconfig.get_build_info())
    except Exception as e:
        print(f"Error running diagnostics: {e}")

logging.basicConfig(level=logging.INFO)

import pandas as pd
from tqdm import tqdm

from audio import save_audio
from birdnet import analyze_audio
from wavelet import wavelet_denoise


def get_data_path():
    """
    Returns the root path for data files
    """
    # You can customize this function based on your project structure
    return os.getenv("DATA_PATH", "/home/mikhail/prj/bc_25_data")


DATA_PATH = get_data_path()


def read_df():
    # Load the dataset
    try:
        csv_path = os.path.join(DATA_PATH, 'train.csv')
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from train.csv")
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None


DF = read_df()


def split_species_safely(species_str):
    """
    Safely split species string into scientific and common name
    Returns tuple of (scientific_name, common_name)
    """
    try:
        if species_str and "_" in species_str:
            parts = species_str.split("_", 1)
            return parts[0], parts[1]
        else:
            return "", ""
    except Exception:
        return "", ""


def main():
    # Check if DataFrame was successfully loaded
    if DF is None:
        print("Error: Could not load DataFrame. Exiting.")
        return

    # Take a random sample of 100 rows
    sample_df = DF.sample(n=100, random_state=random.randint(0, 1000))

    # Initialize counters
    original_correct = 0
    denoised_correct = 0
    total_processed = 0
    errors = 0

    # Create output directory
    output_dir = "denoised_audio"
    os.makedirs(output_dir, exist_ok=True)

    # Paths to BirdNET model files
    tflite = "/home/mikhail/prj/bird_clef_25/bn/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"
    labels_file = "/home/mikhail/prj/bird_clef_25/bn/BirdNET_GLOBAL_6K_V2.4_Labels.txt"

    # Check if files exist
    if not os.path.exists(tflite):
        print(f"Error: Model file not found at {tflite}")
        return
    if not os.path.exists(labels_file):
        print(f"Error: Labels file not found at {labels_file}")
        return

    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Processing audio files"):
        filename = row['filename']
        scientific_name = row['scientific_name']
        common_name = row['common_name']

        # Construct input and output paths
        input_file = os.path.join(get_data_path(), "train_audio", filename)
        output_file = os.path.join(output_dir, os.path.basename(filename))

        if not os.path.exists(input_file):
            print(f"Error: Input file not found at {input_file}")
            continue

        try:
            # Denoise and save the audio
            denoised_file = output_file + ".wav"
            if not os.path.exists(denoised_file):
                sr, denoised = wavelet_denoise(input_file)
                save_audio(denoised, sr, denoised_file, "wav")
            else:
                print("File exists, skipping denoising:", denoised_file)

            # Default values in case no predictions are made
            orig_scientific, orig_common = "", ""
            denoised_scientific, denoised_common = "", ""

            # Analyze original audio
            original_predictions = analyze_audio(input_file, tflite, labels_file)
            if original_predictions and len(original_predictions) > 0:
                top_original = original_predictions[0]
                orig_scientific, orig_common = split_species_safely(top_original.get("species", ""))
            else:
                print(f"No predictions for original file: {filename}")

            # Analyze denoised audio
            denoised_predictions = analyze_audio(denoised_file, tflite, labels_file)
            if denoised_predictions and len(denoised_predictions) > 0:
                top_denoised = denoised_predictions[0]
                denoised_scientific, denoised_common = split_species_safely(top_denoised.get("species", ""))
            else:
                print(f"No predictions for denoised file: {filename}")

            # Clean up temporary file
            if os.path.exists(denoised_file):
                os.remove(denoised_file)

            # Check if predictions are correct (case insensitive comparison)
            if orig_scientific.lower() == scientific_name.lower() or orig_common.lower() == common_name.lower():
                original_correct += 1

            if denoised_scientific.lower() == scientific_name.lower() or denoised_common.lower() == common_name.lower():
                denoised_correct += 1

            total_processed += 1

            # Print detailed result for this sample
            print(f"\nFile: {filename}")
            print(f"True species: {scientific_name} ({common_name})")
            print(f"Original prediction: {orig_scientific} ({orig_common})")
            print(f"Denoised prediction: {denoised_scientific} ({denoised_common})")

        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            errors += 1
            traceback.print_exc()

    # Calculate accuracy
    original_accuracy = original_correct / total_processed if total_processed > 0 else 0
    denoised_accuracy = denoised_correct / total_processed if total_processed > 0 else 0

    # Print results
    print(f"\nResults Summary:")
    print(f"Total files processed: {total_processed}")
    print(f"Files with errors: {errors}")
    print(f"Original file accuracy: {original_accuracy:.2%} ({original_correct}/{total_processed})")
    print(f"Denoised file accuracy: {denoised_accuracy:.2%} ({denoised_correct}/{total_processed})")

    if denoised_accuracy > original_accuracy:
        print(f"Denoising improved accuracy by {denoised_accuracy - original_accuracy:.2%}")
    elif denoised_accuracy < original_accuracy:
        print(f"Denoising reduced accuracy by {original_accuracy - denoised_accuracy:.2%}")
    else:
        print("Denoising had no effect on accuracy")


if __name__ == "__main__":
    main()