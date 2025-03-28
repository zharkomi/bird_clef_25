import json
import os
import csv
import uuid

import pandas as pd
from pathlib import Path
import re

# Enable faulthandler for better debugging
import faulthandler
import os
import random
import traceback

from src.audio import save_audio
from src.birdnet import analyze_audio
from src.wavelet import wavelet_denoise
from src.utils import split_species_safely, calculate_probability

TFILE = "bn/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"
LABELS_FILE = "bn/BirdNET_GLOBAL_6K_V2.4_Labels.txt"


def predict_denoised(workd_dir, input_file):
    """
    Analyze an audio file with wavelet denoising and return all predictions.

    Returns:
        list: List of prediction objects, each containing species, confidence, timestamp, and probability
    """
    # Create output directory
    denoised_file = workd_dir + str(uuid.uuid4()) + ".denoised.wav"

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
    save_audio(denoised, sr, denoised_file, "wav")

    # Analyze denoised audio
    denoised_predictions = analyze_audio(denoised_file, TFILE, LABELS_FILE)
    denoised_predictions = calculate_probability(denoised_predictions)
    print(json.dumps(denoised_predictions, indent=4))

    # Clean up temporary file
    os.remove(denoised_file)

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
def process_audio_file(workd_dir, file_path, scientific_to_id, common_to_id, all_species_ids):
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
        result = {**{'row_id': row_id}, **{id: 0.0 for id in all_species_ids}}

        # Set the probability for the detected species using the confidence from prediction
        result[species_id] = prediction["probability"]  # Using actual probability from prediction

        # Add the row_id to the result

        results.append(result)

    return results


# Main function to process all audio files
def process_all_audio_files(workd_dir, audio_dir, csv_path, output_path):
    # Load species data
    species_df, scientific_to_id, common_to_id = load_species_data(csv_path)

    if species_df is None:
        return

    # Get all unique species IDs
    all_species_ids = species_df['primary_label'].unique().tolist()

    # Prepare results list
    all_results = []

    # Process each audio file
    for filename in os.listdir(audio_dir):
        print("\n------------------------- Predicting for file:", filename)
        if filename.endswith('.ogg') or filename.endswith('.wav'):  # Add other audio formats if needed
            file_path = os.path.join(audio_dir, filename)
            file_results = process_audio_file(workd_dir, file_path, scientific_to_id, common_to_id, all_species_ids)

            if file_results:
                all_results.extend(file_results)

    # Convert results to DataFrame and save
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    else:
        print("No results generated")