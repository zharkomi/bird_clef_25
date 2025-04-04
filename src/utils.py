import os

import soundfile as sf

from src.audio import parse_file
from src.wavelet import wavelet_denoise

TRAIN_DIR = "--path-to-your-train-dir--"


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


def get_best_prediction(predictions):
    """
    Identifies the best prediction based on the longest total duration across all timestamp ranges.
    No merging of overlapping timestamps is performed.

    Args:
        predictions: List of dictionaries with 'species', 'confidence', and 'timestamp' keys
                     where timestamp is in the format 'start-end'

    Returns:
        Dictionary with the species that has the longest total duration and its total duration
    """
    # Dictionary to store total duration for each species
    species_durations = {}

    # Process each prediction - one cycle through the array
    for pred in predictions:
        species = pred['species']
        # Parse the timestamp
        start, end = map(float, pred['timestamp'].split('-'))
        duration = (end - start) * pred['confidence']

        # Add duration to the running total for this species
        if species in species_durations:
            species_durations[species] += duration
        else:
            species_durations[species] = duration

    # Find the species with the longest duration
    best_species = max(species_durations.items(), key=lambda x: x[1])

    return best_species[0]


def load_clef_labels():
    return sorted(os.listdir(TRAIN_DIR))


def denoise_and_play(file_path):
    # Parse audio file
    sr, y = parse_file(file_path)
    _, y1 = wavelet_denoise(sr, y,
                            denoise_method='swt',
                            n_noise_layers=6,
                            wavelet='sym5',
                            level=6,
                            threshold_method='soft',
                            threshold_factor=0.75,
                            denoise_strength=2.0,
                            preserve_ratio=0.2)
    denoised_file_path = file_path + ".denoised.wav"
    sf.write(denoised_file_path, y1, sr, format="wav")

    os.system(f"vlc {file_path} {denoised_file_path}")
