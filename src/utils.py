import pandas as pd
from scipy.special import expit  # This is the logistic sigmoid function


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
