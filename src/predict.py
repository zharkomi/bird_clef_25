import json
# Enable faulthandler for better debugging
import os
import time

import pandas as pd

from src.audio import parse_file
from src.birdnet import analyze_audio
from src.utils import split_species_safely
from src.wavelet import wavelet_denoise

TFILE = "bn/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"
LABELS_FILE = "bn/BirdNET_GLOBAL_6K_V2.4_Labels.txt"


def merge_consecutive_segments(predictions):
    """
    Merge consecutive segments of the same species from BirdNET predictions.

    Args:
        predictions (list): List of prediction objects from analyze_audio, each containing
                           species, confidence, and timestamp.

    Returns:
        list: Merged predictions with:
              - Start time from first segment
              - End time from last segment
              - Max confidence from all merged segments
              - Other fields preserved
    """
    if not predictions:
        return []

    # Sort predictions by species and start time
    sorted_predictions = sorted(predictions, key=lambda x: (
        x["species"],
        float(x["timestamp"].split("-")[0])
    ))

    merged_predictions = []
    current_group = None

    for pred in sorted_predictions:
        # Extract species and time information
        species = pred["species"]
        start_time, end_time = map(float, pred["timestamp"].split("-"))
        confidence = pred["confidence"]

        # If this is a new group or different species
        if (current_group is None or
                current_group["species"] != species):

            # Save the previous group if it exists
            if current_group is not None:
                merged_predictions.append({
                    "species": current_group["species"],
                    "confidence": current_group["max_confidence"],
                    "timestamp": f"{current_group['start_time']:.2f}-{current_group['end_time']:.2f}"
                })

            # Start a new group
            current_group = {
                "species": species,
                "start_time": start_time,
                "end_time": end_time,
                "max_confidence": confidence
            }
        else:
            # Extend the current group and update max confidence
            current_group["end_time"] = max(current_group["end_time"], end_time)
            current_group["max_confidence"] = max(current_group["max_confidence"], confidence)

    # Add the last group
    if current_group is not None:
        merged_predictions.append({
            "species": current_group["species"],
            "confidence": current_group["max_confidence"],
            "timestamp": f"{current_group['start_time']:.2f}-{current_group['end_time']:.2f}"
        })

    return merged_predictions


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
    Analyze an audio file with wavelet denoising and return all predictions.

    Returns:
        list: List of prediction objects, each containing species, confidence, timestamp, and probability
    """
    # Apply wavelet denoising
    sr, denoised = wavelet_denoise(sr, y,
                                   denoise_method=denoise_method,
                                   n_noise_layers=n_noise_layers,
                                   wavelet=wavelet,
                                   level=level,
                                   threshold_method=threshold_method,
                                   threshold_factor=threshold_factor,
                                   denoise_strength=denoise_strength,
                                   preserve_ratio=preserve_ratio
                                   )

    # Analyze denoised audio directly without saving to a file
    denoised_predictions = analyze_audio(denoised, TFILE, LABELS_FILE, sample_rate=sr)
    denoised_predictions = merge_consecutive_segments(denoised_predictions)
    # denoised_predictions = calculate_probability(denoised_predictions)
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


def process_all_audio_files(workd_dir, audio_dir, csv_path, output_path, labels_dir, dir_limit=0):
    """
    Process all audio files from a single directory and create predictions.

    Args:
        workd_dir (str): Working directory
        audio_dir (str): Directory containing audio files
        csv_path (str): Path to the CSV with species information
        output_path (str): Path to save the output CSV
        labels_dir (str): Directory to extract class labels from
        dir_limit (int): Limit on number of files to process (0 = no limit)
    """
    # Load species data
    try:
        species_df, scientific_to_id, common_to_id = load_species_data(csv_path)
        if species_df is None:
            return
    except Exception as e:
        print(f"Error loading species data: {str(e)}. Skipping.")
        return

    # Get class labels (column names) from directory
    try:
        # Extract the class labels from the provided labels directory
        class_labels = sorted(os.listdir(labels_dir))
        expected_columns = ['row_id'] + class_labels
        print(f"Using {len(class_labels)} class labels from directory: {labels_dir}")
    except Exception as e:
        print(f"Error getting class labels from directory {labels_dir}: {str(e)}. Cannot continue.")
        return

    # Prepare results DataFrame
    predictions = pd.DataFrame(columns=expected_columns)

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
        try:
            start_time = time.time()

            print(f"\n-------------- Predicting for file: {file_path}")
            filename = os.path.basename(file_path)

            # Extract soundscape_id from the filename
            soundscape_id = filename.replace(".ogg", "")

            # Parse audio file if existing results weren't found
            sr, y = parse_file(file_path)

            # Get predictions
            predictions_list = predict_denoised(sr, y)

            if not predictions_list or len(predictions_list) == 0:
                print(f"No predictions for file: {file_path}")
                continue

            # Process each prediction
            for prediction in predictions_list:
                # Extract species information
                species_full = prediction["species"]
                scientific_name, common_name = split_species_safely(species_full)

                # Extract timestamp and convert to end time in seconds
                timestamp = prediction["timestamp"]
                end_time = int(float(timestamp.split("-")[1]))

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

                # Initialize new row with all zeros for species probabilities
                new_row_data = {'row_id': row_id}
                for label in class_labels:
                    new_row_data[label] = 0.0

                # Set the probability for the detected species
                if species_id in new_row_data:
                    new_row_data[species_id] = 1.0
                    # Add row to DataFrame
                    new_row = pd.DataFrame([new_row_data])
                    predictions = pd.concat([predictions, new_row], axis=0, ignore_index=True)
                else:
                    print("Species ID not found in expected columns")

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
            print(f"Error processing file {file_path}: {str(e)}. Skipping.")
            continue

    # Calculate average processing time
    if total_files > 0:
        avg_time = total_processing_time / total_files
        print(f"\n------------------------- Processing Summary")
        print(f"Total files processed: {total_files}")
        print(f"Average processing time per file: {avg_time:.2f} seconds")
        print(f"Total processing time: {total_processing_time:.2f} seconds")

    # Save to CSV
    try:
        predictions.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error saving results to {output_path}: {str(e)}")
