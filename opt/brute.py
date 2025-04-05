# Enable faulthandler for better debugging
import csv
import faulthandler
import itertools
import os
import random
import time
import traceback
from datetime import datetime, timedelta
import concurrent.futures

from cuda import setup_cuda
from src import utils
from src.audio import parse_file
from src.new_species import SPECIES
from src.predict import predict_denoised, predict_audio

setup_cuda()

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


def process_file(data):
    """
    Process a single audio file with both original and denoised methods.
    This function will be called by each worker in the parallel pool.

    Args:
        data: A tuple containing (idx, row, denoise_params)

    Returns:
        A tuple of (original_correct, denoised_correct, success_flag, filename)
    """
    idx, row, denoise_params = data
    filename = row['filename']
    primary_label = row['primary_label']

    # Construct input path
    input_file = os.path.join(get_data_path(), "train_audio", filename)

    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        return 0, 0, False, filename

    try:
        # Parse audio file
        sr, y = parse_file(input_file)

        # Process with denoising and get prediction
        _, denoised_predictions = predict_denoised(sr, y, **denoise_params)
        denoised_correct = denoised_predictions.get(primary_label, 0)

        # Analyze original audio
        _, original_predictions = predict_audio(sr, y)
        original_correct = original_predictions.get(primary_label, 0)

        return denoised_correct, original_correct, True, filename

    except Exception as e:
        print(f"Error processing file {filename}: {str(e)}")
        traceback.print_exc()
        return 0, 0, False, filename


def calc_dif(denoise_method,
             n_noise_layers,
             wavelet,
             level,
             threshold_method,
             threshold_factor,
             denoise_strength,
             preserve_ratio,
             max_workers=20):  # New parameter for controlling parallelism
    try:
        # Check if DataFrame was successfully loaded
        if DF is None:
            print("Error: Could not load DataFrame. Exiting.")
            return -10000

        # First filter the DataFrame to exclude rows with row_id in EXCLUDE_LIST
        filtered_df = DF[~DF['primary_label'].isin(SPECIES)]

        # Then sample from the filtered DataFrame
        sample_df = filtered_df.sample(n=max_workers, random_state=random.randint(0, 1000))

        # Create a dictionary of denoising parameters
        denoise_params = {
            'denoise_method': denoise_method,
            'n_noise_layers': n_noise_layers,
            'wavelet': wavelet,
            'level': level,
            'threshold_method': threshold_method,
            'threshold_factor': threshold_factor,
            'denoise_strength': denoise_strength,
            'preserve_ratio': preserve_ratio
        }

        # Prepare data for parallel processing
        process_data = [(idx, row, denoise_params) for idx, row in sample_df.iterrows()]

        # Initialize counters
        original_correct = 0
        denoised_correct = 0
        total_processed = 0
        errors = 0

        # Process files in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks and create a list of futures
            futures = [executor.submit(process_file, data) for data in process_data]

            # Process results with progress bar
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                               desc="Processing audio files"):
                orig, denoised, success, filename = future.result()
                if success:
                    original_correct += orig
                    denoised_correct += denoised
                    total_processed += 1
                    print(f"Processed file: {filename}")
                else:
                    errors += 1

        # Print results
        print(f"Results Summary:")
        print(f"Total files processed: {total_processed}")
        print(f"Files with errors: {errors}")
        print(f"Original file accuracy: {original_correct}")
        print(f"Denoised file accuracy: {denoised_correct}")
        print("")

        return denoised_correct - original_correct

    except Exception as e:
        print("Error in calc_dif:", str(e))
        traceback.print_exc()
        return -10000


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


def brute_force(max_workers=None):
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
            result = calc_dif(**params, max_workers=max_workers)

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
    utils.TRAIN_DIR = "/home/mikhail/prj/bc_25_data/train_audio"
    utils.CSV_PATH = "/home/mikhail/prj/bc_25_data/taxonomy.csv"

    # Use default number of workers (CPU count)
    brute_force()

    # Or specify a specific number of workers
    # brute_force(max_workers=4)