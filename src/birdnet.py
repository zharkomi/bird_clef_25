import librosa
import numpy as np
import pandas as pd
from birdnetlib.analyzer import Analyzer
from birdnetlib import RecordingBuffer

from src import utils
from src.utils import load_clef_labels

# Global variables to store loaded analyzer and labels
_analyzer = None
_labels = None
_species_df = None
_scientific_to_id = {}
_common_to_id = {}

MODEL_PATH = "bn/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"
LABELS_FILE = "bn/BirdNET_GLOBAL_6K_V2.4_Labels.txt"
CSV_PATH = "--path-to-your-csv--"


def get_analyzer():
    """Get or initialize the BirdNET analyzer"""
    global _analyzer
    if _analyzer is None:
        _analyzer = Analyzer()
        _analyzer.model_path = MODEL_PATH
        _analyzer.load_model()
    return _analyzer


def load_birdnet_labels():
    """Load species labels from file."""
    global _labels
    if _labels is None:
        with open(LABELS_FILE, 'r', encoding='utf-8') as f:
            _labels = [line.strip() for line in f]
    return _labels


def load_species_data():
    """
    Load species data from CSV and create mappings for scientific and common names.
    """
    global _species_df, _scientific_to_id, _common_to_id
    if _species_df is None:
        _species_df = pd.read_csv(CSV_PATH)
        # Create a mapping from scientific name to species ID
        _scientific_to_id = dict(zip(_species_df['scientific_name'].str.lower(), _species_df['primary_label']))
        # Create a mapping from common name to species ID (for species that have common names)
        _common_to_id = dict(zip(_species_df['common_name'].str.lower(), _species_df['primary_label']))


def split_species_safely(species_full):
    """
    Split the full species string into scientific name and common name by underscore.

    Parameters:
    - species_full: Full species string in format "Scientific_name_Common_name"

    Returns:
    - Tuple of (scientific_name, common_name)
    """
    # Default values
    scientific_name = species_full
    common_name = ""

    # Split by underscore if present
    if "_" in species_full:
        parts = species_full.split("_", 1)  # Split on first underscore only
        scientific_name = parts[0].strip()
        if len(parts) > 1:
            common_name = parts[1].strip()

    return scientific_name.lower(), common_name.lower()


def analyze_audio_fixed_chunks(audio,
                               chunk_duration=5,
                               sample_rate=32000):
    """
    Analyze bird sounds in fixed, non-overlapping chunks and map to required class labels.
    Uses birdnetlib for all processing.

    Parameters:
    - audio: Audio signal as numpy array
    - chunk_duration: Duration of each chunk in seconds (default: 5)
    - sample_rate: Sample rate of the audio

    Returns:
    - List of dictionaries, one for each chunk. Each dictionary contains:
      - 'row_id': The end time of the chunk in seconds
      - And one key for each species_id with its confidence score
    """
    # Load analyzer and labels
    analyzer = get_analyzer()
    labels = load_birdnet_labels()
    class_labels = load_clef_labels()

    # Load species data if not already loaded
    load_species_data()

    # Ensure correct sample rate for BirdNET
    birdnet_sr = 48000
    if sample_rate != birdnet_sr:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=birdnet_sr)
        sample_rate = birdnet_sr

    # Calculate chunk size in samples
    chunk_size = int(chunk_duration * sample_rate)

    # Prepare results list
    results = []

    # Process audio in non-overlapping chunks
    for chunk_idx, chunk_start in enumerate(range(0, len(audio), chunk_size)):
        # Extract chunk
        chunk_end = min(chunk_start + chunk_size, len(audio))
        chunk = audio[chunk_start:chunk_end]

        # Initialize probabilities for all class_labels (set to 0 initially)
        chunk_result = {species_id: 0.0 for species_id in class_labels}

        # Skip prediction if chunk is too short
        if len(chunk) < sample_rate:
            chunk_result['row_id'] = chunk_end // sample_rate
            results.append(chunk_result)
            continue

        # Normalize chunk (with safety check for silent segments)
        max_abs_val = np.max(np.abs(chunk))
        if max_abs_val > 0:
            chunk = chunk / max_abs_val

        # Create a RecordingBuffer object for this chunk
        # RecordingBuffer is designed to work with raw audio buffers
        temp_recording = RecordingBuffer(
            analyzer,
            buffer=chunk,
            rate=sample_rate,
            return_all_detections=True
        )

        # Process the audio data
        temp_recording.analyze()

        # Get predictions
        predictions = temp_recording.detections

        # Map birdnetlib predictions to our class labels
        # analyze() returns a list of dictionaries with detection details
        for detection in predictions:

            # Get both scientific and common names from detection
            scientific_name = detection['scientific_name'].lower() if 'scientific_name' in detection else ""
            common_name = detection['common_name'].lower() if 'common_name' in detection else ""
            confidence = detection['confidence']

            # Find matching species ID
            species_id = None
            if scientific_name and scientific_name in _scientific_to_id:
                species_id = _scientific_to_id[scientific_name]
            elif common_name and common_name in _common_to_id:
                species_id = _common_to_id[common_name]

            # Update probability if species is in our class_labels
            if species_id is not None and species_id in class_labels:
                chunk_result[species_id] = confidence

        # Add row_id to the chunk result
        chunk_result['row_id'] = chunk_end // sample_rate

        # Add result for this chunk
        results.append(chunk_result)

    return results
