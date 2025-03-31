import os
import threading
import time
import gc
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import pandas as pd
import tensorflow as tf

from src.audio import parse_file
from src.birdnet import analyze_audio_fixed_chunks
from src.birdnet import load_species_data
from src.utils import load_clef_labels
from src.wavelet import wavelet_denoise


def count_species_occurrences(predictions, threshold=0.5):
    """
    Count occurrences of each species above a threshold.

    Args:
        predictions: List of dictionaries where each dictionary contains species names as keys
                    and confidence scores as values.
        threshold: Minimum confidence score to count an occurrence (default: 0.5).

    Returns:
        dict: Dictionary mapping species names to their occurrence counts above the threshold.
    """
    species_counts = {}

    # Iterate through each prediction dictionary
    for prediction in predictions:
        # Count keys with values greater than the threshold
        for species, confidence in prediction.items():
            if confidence > threshold:
                # Increment the count for this species
                if species in species_counts:
                    species_counts[species] += 1
                else:
                    species_counts[species] = 1

    return species_counts


def predict_denoised(sr, y,
                     denoise_method='emd',
                     n_noise_layers=3,
                     wavelet='db8',
                     level=5,
                     threshold_method='soft',
                     threshold_factor=0.9,
                     denoise_strength=2.0,
                     preserve_ratio=0.9
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
    # Process denoised audio with fixed chunk analysis
    predictions = analyze_audio_fixed_chunks(
        audio,
        chunk_duration=5,
        sample_rate=sr
    )
    print(f"Generated {len(predictions)} chunk predictions")
    # Count species occurrences and print results
    species_counts = count_species_occurrences(predictions, threshold=0.5)
    print("Species occurrences with probability > 0.5:")
    print(species_counts)
    return predictions, species_counts


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
            chunk_predictions, _ = predict_denoised(sr, y)

            # Create a deep copy of the results to avoid TensorFlow references
            file_results = []
            for chunk_result in chunk_predictions:
                # Create a new dictionary with copies of all values
                safe_result = dict(chunk_result)  # Create a copy
                safe_result['row_id'] = f"{soundscape_id}_{safe_result['row_id']}"
                file_results.append(safe_result)

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


def process_all_audio_files(audio_dir, output_path, batch_size=4, dir_limit=0,
                            use_multiprocessing=False):
    """
    Process all audio files from a single directory in parallel batches and create predictions.

    Args:
        audio_dir (str): Directory containing audio files
        output_path (str): Path to save the output CSV
        batch_size (int): Number of files to process in each batch
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
    print(
        f"Processing in batches of {batch_size} files parallel {'processes' if use_multiprocessing else 'threads'}")

    # Process files in batches
    total_processing_time = 0
    total_files_processed = 0

    # Split files into batches
    batches = [test_soundscapes[i:i + batch_size] for i in range(0, len(test_soundscapes), batch_size)]

    for batch_num, batch in enumerate(batches):
        batch_start_time = time.time()
        print(f"\n------------------------- Processing batch {batch_num + 1}/{len(batches)}")

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

        total_files_processed += files_processed_in_batch

        # Add batch results to the main DataFrame with lock
        with results_lock:
            for result in batch_results:
                new_row = pd.DataFrame([result])
                predictions_df = pd.concat([predictions_df, new_row], axis=0, ignore_index=True)

        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time
        total_processing_time += batch_time

        print(f"Batch {batch_num + 1} processing time: {batch_time:.2f} seconds")
        print(f"Added {len(batch_results)} predictions to DataFrame")

        # Log progress but don't create intermediate files
        progress_percentage = (batch_num + 1) / len(batches) * 100
        print(f"Progress: {progress_percentage:.1f}% complete ({batch_num + 1}/{len(batches)} batches)")

        # Clear any lingering TensorFlow state between batches
        tf.keras.backend.clear_session()
        gc.collect()

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