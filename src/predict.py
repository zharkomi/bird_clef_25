import gc
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import pandas as pd
import tensorflow as tf

from src.audio import parse_file
from src.birdnet import analyze_audio
from src.birdnet import load_species_data
from src.utils import load_clef_labels
from src.wavelet import wavelet_denoise


def calc_species_stat(predictions):
    """
    Count occurrences of each species above a threshold.

    Args:
        predictions: List of dictionaries where each dictionary contains species names as keys
                    and confidence scores as values.

    Returns:
        dict: Dictionary mapping species names to their occurrence counts above the threshold.
    """
    species_counts = {}

    # Iterate through each prediction dictionary
    for prediction in predictions:
        # Count keys with values greater than the threshold
        for species, confidence in prediction.items():
            if isinstance(confidence, float):
                species_counts[species] = species_counts.get(species, 0.0) + confidence

    return species_counts


def predict_denoised(sr, y,
                     denoise_method='dwt',
                     n_noise_layers=3,
                     wavelet='coif3',
                     level=2,
                     threshold_method='soft',
                     threshold_factor=0.65,
                     denoise_strength=0.73,
                     preserve_ratio=0.83
                     ):
    """
    Analyze an audio file with wavelet denoising using fixed-chunk processing.

    Args:
        sr: Sample rate
        y: Audio data
        denoise_method: Method for denoising ('emd', 'wavelet', etc.)
        n_noise_layers: Number of noise layers for EMD denoising
        wavelet: Wavelet type for denoising
        level: Decomposition level for wavelet denoising
        threshold_method: Threshold method for wavelet denoising
        threshold_factor: Threshold factor for wavelet denoising
        denoise_strength: Strength of denoising
        preserve_ratio: Ratio of signal to preserve during denoising

    Returns:
        list: List of dictionaries, one for each chunk, containing chunk_id, time_end,
              and probabilities for each species in class_labels
    """
    # Apply wavelet denoising
    sr, denoised = wavelet_denoise(sr, y, denoise_method=denoise_method, n_noise_layers=n_noise_layers, wavelet=wavelet,
                                   level=level, threshold_method=threshold_method, threshold_factor=threshold_factor,
                                   denoise_strength=denoise_strength, preserve_ratio=preserve_ratio)
    return predict_audio(sr, denoised)


def predict_audio(sr, audio):
    predictions = analyze_audio(
        audio,
        sample_rate=sr
    )
    print(f"Generated {len(predictions)} chunk predictions")
    # Count species occurrences and print results
    species_stat = calc_species_stat(predictions)
    print("Species occurrences:")
    print(species_stat)
    return predictions, species_stat


# Function to process a single audio file
def process_audio_file(file_path):
    try:
        # Create a new TensorFlow graph for this thread
        with tf.Graph().as_default():
            start_time = time.time()

            print(f"\n-------------- Predicting for file: {file_path}")
            filename = os.path.basename(file_path)

            # Extract soundscape_id from the filename
            soundscape_id = filename.replace(".ogg", "")

            # Parse audio file
            sr, y = parse_file(file_path)

            # Get predictions using the fixed-chunk method
            # chunk_predictions, _ = predict_denoised(sr, y)
            chunk_predictions, _ = predict_audio(sr, y)

            # Create a deep copy of the results to avoid TensorFlow references
            file_results = []
            for chunk_result in chunk_predictions:
                chunk_result['row_id'] = f"{soundscape_id}_{chunk_result['row_id']}"
                file_results.append(chunk_result)

            # Calculate processing time
            end_time = time.time()
            elapsed_time = end_time - start_time

            print(f"Processing time for {filename}: {elapsed_time:.2f} seconds")
            print(f"Generated {len(chunk_predictions)} chunk predictions")

            gc.collect()

            return file_results
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}. Skipping.")
        traceback.print_exc()
        # Try to clean up anyway
        try:
            tf.keras.backend.clear_session()
            gc.collect()
        except:
            pass
        return []


def process_audio_batch(batch, batch_num, total_batches, batch_size, predictions_df, results_lock,
                        use_multiprocessing=False):
    """
    Process a batch of audio files in parallel and add results to the predictions DataFrame.

    Args:
        batch (list): List of file paths to process
        batch_num (int): Current batch number
        total_batches (int): Total number of batches
        batch_size (int): Size of the batch
        predictions_df (pd.DataFrame): DataFrame to store results
        results_lock (threading.Lock): Lock for thread-safe DataFrame updates
        use_multiprocessing (bool): Whether to use multiprocessing instead of threading

    Returns:
        tuple: (batch_time, files_processed_in_batch, updated_df)
    """
    batch_start_time = time.time()
    print(f"\n------------------------- Processing batch {batch_num + 1}/{total_batches}")

    batch_results = []
    files_processed_in_batch = 0

    # Choose executor based on preference (ProcessPoolExecutor provides better isolation)
    ExecutorClass = ProcessPoolExecutor if use_multiprocessing else ThreadPoolExecutor

    # Process batch in parallel
    with ExecutorClass(max_workers=batch_size) as executor:
        # Submit all files in this batch for processing
        future_to_file = {executor.submit(process_audio_file, file_path): file_path for file_path in batch}

        # Collect results as they complete
        for future in future_to_file:
            try:
                file_results = future.result()
                if file_results:
                    batch_results.extend(file_results)
                    files_processed_in_batch += 1
            except Exception as e:
                print(f"Error in parallel execution: {str(e)}")

            # Force garbage collection after each file to free memory
            gc.collect()

    # Create updated DataFrame with the new results
    updated_df = predictions_df
    if batch_results:
        batch_df = pd.DataFrame(batch_results)
        # Only update the DataFrame if there are results
        with results_lock:
            updated_df = pd.concat([predictions_df, batch_df], axis=0, ignore_index=True)

    batch_end_time = time.time()
    batch_time = batch_end_time - batch_start_time

    print(f"Batch {batch_num + 1} processing time: {batch_time:.2f} seconds")
    print(f"Added {len(batch_results)} predictions to DataFrame")

    # Log progress
    progress_percentage = (batch_num + 1) / total_batches * 100
    print(f"Progress: {progress_percentage:.1f}% complete ({batch_num + 1}/{total_batches} batches)")

    # Clear any lingering TensorFlow state between batches
    tf.keras.backend.clear_session()
    gc.collect()

    return batch_time, files_processed_in_batch, updated_df


def process_audio_sequentially(files, predictions_df, results_lock):
    """
    Process audio files sequentially, one at a time.

    Args:
        files (list): List of file paths to process
        predictions_df (pd.DataFrame): DataFrame to store results
        results_lock (threading.Lock): Lock for thread-safe DataFrame updates

    Returns:
        tuple: (total_processing_time, total_files_processed, updated_df)
    """
    total_processing_time = 0
    total_files_processed = 0
    updated_df = predictions_df

    print(f"\n------------------------- Processing {len(files)} files sequentially")

    for i, file_path in enumerate(files):
        file_start_time = time.time()

        try:
            file_results = process_audio_file(file_path)

            if file_results:
                # Add results to the DataFrame
                file_df = pd.DataFrame(file_results)
                with results_lock:
                    updated_df = pd.concat([updated_df, file_df], axis=0, ignore_index=True)

                total_files_processed += 1

            file_end_time = time.time()
            file_time = file_end_time - file_start_time
            total_processing_time += file_time

            print(f"File {i + 1}/{len(files)} processing time: {file_time:.2f} seconds")

            # Log progress
            progress_percentage = (i + 1) / len(files) * 100
            print(f"Progress: {progress_percentage:.1f}% complete ({i + 1}/{len(files)} files)")

        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")

        # Clear TensorFlow state between files
        tf.keras.backend.clear_session()
        gc.collect()

    return total_processing_time, total_files_processed, updated_df


def process_all_audio_files(audio_dir, output_path, batch_size=-1, dir_limit=0,
                            use_multiprocessing=False):
    """
    Process all audio files from a single directory and create predictions.

    Args:
        audio_dir (str): Directory containing audio files
        output_path (str): Path to save the output CSV
        batch_size (int): Number of files to process in each batch (0 or less for sequential processing)
        dir_limit (int): Limit on number of files to process (0 = no limit)
        use_multiprocessing (bool): Use multiprocessing instead of threading (better isolation for TensorFlow)
    """
    # Configure TensorFlow for better multiprocessing/threading behavior
    # Limit TensorFlow's resource usage
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Limit memory growth to avoid OOM errors
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled for {len(gpus)} GPUs")
        except RuntimeError as e:
            print(f"Error setting GPU memory growth: {e}")

    # Configure threading behavior
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    # Load required data
    load_species_data()

    # Get class labels (column names) from directory
    class_labels = load_clef_labels()
    expected_columns = ['row_id'] + class_labels

    # Prepare results DataFrame
    predictions_df = pd.DataFrame(columns=expected_columns)
    results_lock = threading.Lock()  # Lock for thread-safe DataFrame updates

    # Make sure the directory exists
    if not os.path.isdir(audio_dir):
        print(f"Warning: Directory {audio_dir} does not exist. Skipping.")
        return

    # Get list of audio files
    try:
        test_soundscapes = [os.path.join(audio_dir, afile) for afile in
                            sorted(os.listdir(audio_dir)) if afile.endswith('.ogg')]
    except Exception as e:
        print(f"Error listing directory {audio_dir}: {str(e)}. Skipping.")
        return

    # Apply limit if specified
    if 0 < dir_limit < len(test_soundscapes):
        test_soundscapes = test_soundscapes[:dir_limit]

    print(f"\n------------------------- Processing directory: {audio_dir}")
    print(f"Found {len(test_soundscapes)} audio files to process")

    total_processing_time = 0
    total_files_processed = 0

    if batch_size <= 0:
        # Process files sequentially
        print(f"Processing files sequentially")
        total_processing_time, total_files_processed, predictions_df = process_audio_sequentially(
            test_soundscapes, predictions_df, results_lock
        )
    else:
        # Process files in parallel batches
        print(
            f"Processing in batches of {batch_size} files parallel {'processes' if use_multiprocessing else 'threads'}")

        # Split files into batches
        batches = [test_soundscapes[i:i + batch_size] for i in range(0, len(test_soundscapes), batch_size)]

        for batch_num, batch in enumerate(batches):
            batch_time, files_processed, predictions_df = process_audio_batch(
                batch, batch_num, len(batches), batch_size,
                predictions_df, results_lock, use_multiprocessing
            )

            total_processing_time += batch_time
            total_files_processed += files_processed

    # Calculate and print summary
    if total_files_processed > 0:
        avg_time = total_processing_time / total_files_processed
        print(f"\n------------------------- Processing Summary")
        print(f"Total files processed: {total_files_processed}")
        print(f"Average processing time per file: {avg_time:.2f} seconds")
        print(f"Total processing time: {total_processing_time:.2f} seconds")
        print(f"Total prediction rows: {len(predictions_df)}")

    # Save final results to CSV
    try:
        predictions_df.to_csv(output_path, index=False)
        print(f"Final results saved to {output_path}")
    except Exception as e:
        print(f"Error saving results to {output_path}: {str(e)}")

    return predictions_df
