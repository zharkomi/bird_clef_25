import json
import os
import time
import pandas as pd

from src import utils
from src.audio import parse_file
from src.birdnet import analyze_audio_fixed_chunks, load_species_data
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
        species_csv_path: Path to species CSV mapping file
        class_labels: List of class labels required for prediction
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


def process_all_audio_files(audio_dir, output_path, dir_limit=0):
    """
    Process all audio files from a single directory and create predictions.

    Args:
        audio_dir (str): Directory containing audio files
        output_path (str): Path to save the output CSV
        dir_limit (int): Limit on number of files to process (0 = no limit)
    """
    load_species_data()

    # Get class labels (column names) from directory
    # Extract the class labels from the provided labels directory
    class_labels = load_clef_labels()
    expected_columns = ['row_id'] + class_labels

    # Prepare results DataFrame
    predictions_df = pd.DataFrame(columns=expected_columns)

    # Make sure the directory exists
    if not os.path.isdir(audio_dir):
        print(f"Warning: Directory {audio_dir} does not exist. Skipping.")
        return

    # Tracking time metrics
    total_processing_time = 0
    total_files = 0
    count = 0

    try:
        # Process each audio file in the directory
        test_soundscapes = [os.path.join(audio_dir, afile) for afile in
                            sorted(os.listdir(audio_dir)) if afile.endswith('.ogg')]
    except Exception as e:
        print(f"Error listing directory {audio_dir}: {str(e)}. Skipping.")
        return

    print(f"\n------------------------- Processing directory: {audio_dir}")
    print(f"Found {len(test_soundscapes)} audio files to process")

    for file_path in test_soundscapes:
        if 0 < dir_limit <= count:
            break
        try:
            start_time = time.time()

            print(f"\n-------------- Predicting for file: {file_path}")
            filename = os.path.basename(file_path)

            # Extract soundscape_id from the filename
            soundscape_id = filename.replace(".ogg", "")

            # Parse audio file
            sr, y = parse_file(file_path)

            # Get predictions using the new fixed-chunk method
            # Now returns a list of dictionaries (one per chunk)
            chunk_predictions, _ = predict_denoised(sr, y)

            # Process each chunk prediction
            for chunk_result in chunk_predictions:
                # Extract time_end from the chunk result
                chunk_result['row_id'] = f"{soundscape_id}_{chunk_result['row_id']}"
                # Add row to DataFrame
                new_row = pd.DataFrame([chunk_result])
                predictions_df = pd.concat([predictions_df, new_row], axis=0, ignore_index=True)

            # Calculate and record processing time
            end_time = time.time()
            elapsed_time = end_time - start_time
            total_processing_time += elapsed_time

            print(f"Processing time for {filename}: {elapsed_time:.2f} seconds")
            print(f"Added {len(chunk_predictions)} rows to predictions DataFrame")

            total_files += 1
            count += 1
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}. Skipping.")
            continue

    # Calculate average processing time
    if total_files > 0:
        avg_time = total_processing_time / total_files
        print(f"\n------------------------- Processing Summary")
        print(f"Total files processed: {total_files}")
        print(f"Average processing time per file: {avg_time:.2f} seconds")
        print(f"Total processing time: {total_processing_time:.2f} seconds")
        print(f"Total prediction rows: {len(predictions_df)}")

    # Save to CSV
    try:
        predictions_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error saving results to {output_path}: {str(e)}")
