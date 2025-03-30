# Enable faulthandler for better debugging
import csv
import faulthandler
import itertools
import os
import random
import time
import traceback
from datetime import datetime, timedelta

from src.audio import parse_file
from src.predict import predict_denoised

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

from src.birdnet import analyze_audio
from src.utils import get_best_prediction, split_species_safely


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

TFILE = "bn/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"
LABELS_FILE = "bn/BirdNET_GLOBAL_6K_V2.4_Labels.txt"


def calc_dif(denoise_method='emd',
             n_noise_layers=3,
             wavelet='db8',
             level=5,
             threshold_method='soft',
             threshold_factor=0.9,
             denoise_strength=2.0,
             preserve_ratio=0.9):
    # Check if DataFrame was successfully loaded
    if DF is None:
        print("Error: Could not load DataFrame. Exiting.")
        return

    # Take a random sample of 100 rows
    sample_df = DF.sample(n=20, random_state=random.randint(0, 1000))

    # Initialize counters
    original_correct = 0
    denoised_correct = 0
    total_processed = 0
    errors = 0

    # Check if files exist
    if not os.path.exists(TFILE):
        print(f"Error: Model file not found at {TFILE}")
        return
    if not os.path.exists(LABELS_FILE):
        print(f"Error: Labels file not found at {LABELS_FILE}")
        return

    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Processing audio files"):
        filename = row['filename']
        scientific_name = row['scientific_name']
        common_name = row['common_name']

        # Construct input path
        input_file = os.path.join(get_data_path(), "train_audio", filename)

        if not os.path.exists(input_file):
            print(f"Error: Input file not found at {input_file}")
            continue

        try:
            # Default values in case no predictions are made
            orig_scientific, orig_common = "", ""
            denoised_common, denoised_scientific = "", ""

            # Parse audio file if existing results weren't found
            sr, y = parse_file(input_file)

            # Process with denoising and get prediction
            denoised_predictions = predict_denoised(sr, y,
                                                    denoise_method=denoise_method,
                                                    n_noise_layers=n_noise_layers,
                                                    wavelet=wavelet,
                                                    level=level,
                                                    threshold_method=threshold_method,
                                                    threshold_factor=threshold_factor,
                                                    denoise_strength=denoise_strength,
                                                    preserve_ratio=preserve_ratio
                                                    )
            if denoised_predictions and len(denoised_predictions) > 0:
                prediction = get_best_prediction(denoised_predictions)
                denoised_scientific, denoised_common = split_species_safely(prediction)
            else:
                print(f"No predictions for denoised file: {filename}")

            # Analyze original audio
            original_predictions = analyze_audio(y, TFILE, LABELS_FILE, sr=sr)
            if original_predictions and len(original_predictions) > 0:
                prediction = get_best_prediction(original_predictions)
                orig_scientific, orig_common = split_species_safely(prediction)
            else:
                print(f"No predictions for original file: {filename}")

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

    return denoised_correct - original_correct


# Define parameter search space
param_map = {
    'denoise_method': ['dwt', 'wpt', 'emd'],
    'n_noise_layers': [1, 3, 6],
    # 'wavelet': ['dmey', 'db6', 'db4', 'db8', 'sym8', 'coif3'],
    'wavelet': ['bior4.4', 'dmey'],
    'level': [1, 3, 6],
    'threshold_method': ['soft', 'hard'],
    'threshold_factor': [0.1, 1.0, 2.0],
    'denoise_strength': [0.5],
    'preserve_ratio': [0.9]
}


def brute_force():
    # Calculate total number of iterations
    total_iterations = 1
    for param_values in param_map.values():
        total_iterations *= len(param_values)

    print(f"Total number of iterations: {total_iterations}")

    # Generate all parameter combinations
    param_keys = list(param_map.keys())
    param_values = list(param_map.values())
    combinations = list(itertools.product(*param_values))

    # Prepare results storage
    results = []

    # Start timing
    start_time = time.time()
    last_update_time = start_time

    # Create output CSV file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"optimization_results_{timestamp}.csv"

    with open(output_filename, 'w', newline='') as csvfile:
        # Create CSV writer
        csv_writer = csv.writer(csvfile)

        # Write header row
        header = param_keys + ['result', 'execution_time']
        csv_writer.writerow(header)

        # Process all combinations
        for i, combo in enumerate(combinations):
            # Create parameter dictionary for this combination
            params = dict(zip(param_keys, combo))

            # Time this specific iteration
            iter_start = time.time()

            # Calculate result for this parameter set
            result = calc_dif(**params)

            # Calculate iteration time
            iter_time = time.time() - iter_start

            # Add to results
            row = list(combo) + [result, iter_time]
            csv_writer.writerow(row)
            results.append((params, result, iter_time))

            # Update progress every 100 iterations or 5 seconds
            current_time = time.time()
            if i % 100 == 0 or current_time - last_update_time > 5:
                # Calculate progress and estimate remaining time
                progress = (i + 1) / total_iterations
                elapsed = current_time - start_time
                if progress > 0:
                    estimated_total = elapsed / progress
                    remaining = estimated_total - elapsed
                    eta = datetime.now() + timedelta(seconds=remaining)

                    # Print progress update
                    print(f"Progress: {i + 1}/{total_iterations} ({progress:.1%})")
                    print(f"Elapsed time: {timedelta(seconds=int(elapsed))}")
                    print(f"Estimated remaining: {timedelta(seconds=int(remaining))}")
                    print(f"ETA: {eta.strftime('%H:%M:%S')}")
                    print(f"Average iteration time: {elapsed / (i + 1):.4f} seconds")
                    print("---")

                    # Flush the CSV to disk
                    csvfile.flush()

                    # Update last update time
                    last_update_time = current_time

    # Find the best result
    best_result = max(results, key=lambda x: x[1])
    best_params, best_value, _ = best_result

    print("\nOptimization complete!")
    print(f"Results saved to: {output_filename}")
    print(f"Best result: {best_value}")
    print("Best parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # Total execution time
    total_time = time.time() - start_time
    print(f"Total execution time: {timedelta(seconds=int(total_time))}")

    return best_params, best_value


if __name__ == "__main__":
    brute_force()
