import json
# Enable faulthandler for better debugging
import os
import time
import uuid

import pandas as pd

from src.audio import save_audio
from src.birdnet import analyze_audio
from src.utils import split_species_safely, calculate_probability
from src.wavelet import wavelet_denoise

TFILE = "bn/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"
LABELS_FILE = "bn/BirdNET_GLOBAL_6K_V2.4_Labels.txt"


def predict_denoised(workd_dir, input_file):
    """
    Analyze an audio file with wavelet denoising and return all predictions.

    Returns:
        list: List of prediction objects, each containing species, confidence, timestamp, and probability
    """
    # Apply wavelet denoising
    sr, denoised = wavelet_denoise(input_file,
                                   denoise_method='emd',
                                   n_noise_layers=3,
                                   wavelet='db8',
                                   level=5,
                                   threshold_method='soft',
                                   threshold_factor=0.9,
                                   denoise_strength=2.0,
                                   preserve_ratio=0.9
                                   )

    # Analyze denoised audio directly without saving to a file
    denoised_predictions = analyze_audio(denoised, TFILE, LABELS_FILE, sample_rate=sr)
    denoised_predictions = calculate_probability(denoised_predictions)
    print(json.dumps(denoised_predictions, indent=4))

    # Return all predictions
    return denoised_predictions


# Load the CSV with species information
def load_species_data(csv_path):
    try:
        species_df = pd.read_csv(csv_path)
        # Create a mapping from scientific name to species ID
        scientific_to_id = dict(zip(species_df['scientific_name'], species_df['primary_label']))
        # Create a mapping from common name to species ID (for species that have common names)
        common_to_id = dict(zip(species_df['common_name'], species_df['primary_label']))
        return species_df, scientific_to_id, common_to_id
    except Exception as e:
        print(f"Error loading species data: {e}")
        return None, {}, {}


# Process a single audio file
def process_audio_file(workd_dir, file_path, scientific_to_id, common_to_id, expected_columns):
    # Extract soundscape_id from the filename
    filename = os.path.basename(file_path)
    soundscape_id = filename.replace(".ogg", "")  # Adjust this extraction based on your actual filename pattern

    # Get predictions
    predictions = predict_denoised(workd_dir, file_path)

    if not predictions or len(predictions) == 0:
        print(f"No predictions for file: {file_path}")
        return []

    results = []

    # Process each prediction
    for prediction in predictions:
        # Extract species information
        species_full = prediction["species"]
        scientific_name, common_name = split_species_safely(species_full)

        # Extract timestamp and convert to end time in seconds
        timestamp = prediction["timestamp"]
        end_time = int(timestamp.split("-")[1].split(".")[0])

        # Find the species ID
        species_id = None

        # Try to match by scientific name first
        if scientific_name in scientific_to_id:
            species_id = scientific_to_id[scientific_name]
        # If no match, try common name
        elif common_name in common_to_id:
            species_id = common_to_id[common_name]

        if species_id is None:
            print(f"No matching species ID found for {scientific_name} / {common_name}")
            continue
        else:
            print(species_id + ": " + scientific_name + ", " + common_name)

        # Create the row_id
        row_id = f"{soundscape_id}_{end_time}"

        # Initialize all species probabilities to 0
        result = {**{'row_id': row_id}, **{id: 0.0 for id in expected_columns if id != 'row_id'}}

        # Set the probability for the detected species using the confidence from prediction
        if species_id in result:
            result[species_id] = prediction["probability"]  # Using actual probability from prediction

        results.append(result)

    return results


# Load sample submission file to get expected structure
def load_sample_submission(sample_path):
    try:
        sample_df = pd.read_csv(sample_path)
        return sample_df
    except Exception as e:
        print(f"Error loading sample submission file: {e}")
        return None


# Main function to process all audio files from multiple directories
def process_all_audio_files(workd_dir, audio_dirs, csv_path, output_path, sample_submission_path, dir_limit=0):
    # Load species data
    try:
        species_df, scientific_to_id, common_to_id = load_species_data(csv_path)
        if species_df is None:
            return
    except Exception as e:
        print(f"Error loading species data: {str(e)}. Skipping.")
        return

    # Load sample submission to get expected structure
    try:
        sample_df = load_sample_submission(sample_submission_path)
        if sample_df is None:
            print("Cannot continue without sample submission file.")
            return
    except Exception as e:
        print(f"Error loading sample submission: {str(e)}. Cannot continue.")
        return

    # Get expected columns from sample submission
    expected_columns = sample_df.columns.tolist()
    print(f"Expected columns from sample submission: {expected_columns}")

    # Prepare results list
    all_results = []

    # Tracking time metrics
    total_processing_time = 0
    total_files = 0

    # Process each directory in the audio_dirs array
    for audio_dir in audio_dirs:
        print(f"\n------------------------- Processing directory: {audio_dir}")

        # Make sure the directory exists
        if not os.path.isdir(audio_dir):
            print(f"Warning: Directory {audio_dir} does not exist. Skipping.")
            continue

        count = 0
        try:
            # Process each audio file in the current directory
            test_soundscapes = [os.path.join(audio_dir, afile) for afile in
                                sorted(os.listdir(audio_dir)) if afile.endswith('.ogg')]
        except Exception as e:
            print(f"Error listing directory {audio_dir}: {str(e)}. Skipping.")
            continue

        for filename in test_soundscapes:
            try:
                if filename.endswith('.ogg'):  # Add other audio formats if needed
                    start_time = time.time()

                    print(f"\n-------------- Predicting for file: {filename}")
                    file_path = os.path.join(audio_dir, filename)
                    file_results = process_audio_file(workd_dir, file_path, scientific_to_id, common_to_id,
                                                      expected_columns)

                    if file_results:
                        all_results.extend(file_results)

                    # Calculate and record processing time
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    total_processing_time += elapsed_time

                    print(f"Processing time for {filename}: {elapsed_time:.2f} seconds")

                    total_files += 1
                    count += 1
                    if 0 < dir_limit <= count:
                        break
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}. Skipping.")
                continue

    # Calculate average processing time
    if total_files > 0:
        avg_time = total_processing_time / total_files
        print(f"\n------------------------- Processing Summary")
        print(f"Total files processed: {total_files}")
        print(f"Average processing time per file: {avg_time:.2f} seconds")
        print(f"Total processing time: {total_processing_time:.2f} seconds")

    # Convert results to DataFrame and save
    try:
        if all_results:
            # Create DataFrame with all results
            results_df = pd.DataFrame(all_results)

            # Ensure all expected columns are present
            for col in expected_columns:
                if col not in results_df.columns:
                    results_df[col] = 0.0

            # Reorder columns to match sample submission
            results_df = results_df[expected_columns]

            # Save to CSV
            results_df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        else:
            # If no results, create an empty DataFrame with the expected structure
            empty_df = pd.DataFrame(columns=expected_columns)
            empty_df.to_csv(output_path, index=False)
            print(f"No results generated. Empty file with expected structure created at {output_path}")
    except Exception as e:
        print(f"Error saving results to {output_path}: {str(e)}")