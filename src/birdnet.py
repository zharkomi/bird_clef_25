import librosa
import numpy as np
import pandas as pd
from birdnetlib import RecordingBuffer
from birdnetlib.analyzer import Analyzer
from scipy.interpolate import CubicSpline

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
        # _analyzer.model_path = MODEL_PATH
        # _analyzer.load_model()
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

        # Get predictions
        predictions = analyze_chunk(chunk, sample_rate).detections

        # Map birdnetlib predictions to our class labels
        # analyze() returns a list of dictionaries with detection details
        for detection in predictions:

            # Get both scientific and common names from detection
            scientific_name = detection['scientific_name'].lower() if 'scientific_name' in detection else ""
            common_name = detection['common_name'].lower() if 'common_name' in detection else ""
            confidence = detection['confidence']

            # Find matching species ID
            species_id = get_species_id(common_name, scientific_name)

            # Update probability if species is in our class_labels
            if species_id is not None and species_id in class_labels:
                chunk_result[species_id] = confidence

        # Add row_id to the chunk result
        chunk_result['row_id'] = chunk_end // sample_rate

        # Add result for this chunk
        results.append(chunk_result)

    return results


def get_species_id(common_name, scientific_name):
    load_species_data()
    species_id = None
    if scientific_name and scientific_name.lower() in _scientific_to_id:
        species_id = _scientific_to_id[scientific_name.lower()]
    elif common_name and common_name.lower() in _common_to_id:
        species_id = _common_to_id[common_name.lower()]
    return species_id


def analyze_chunk(chunk, sample_rate):
    # Create a RecordingBuffer object for this chunk
    # RecordingBuffer is designed to work with raw audio buffers
    analyzer = get_analyzer()
    recording = RecordingBuffer(
        analyzer,
        buffer=chunk,
        rate=sample_rate
    )
    # Process the audio data
    recording.analyze()
    return recording


def analyze_audio(audio, sample_rate):
    """
    Analyze audio and return a list of dictionaries with predictions for bird species.
    Always returns a number of rows equal to the number of 5-second chunks in the audio.

    Parameters:
    - audio: Audio signal as numpy array
    - sample_rate: Sample rate of the audio (default: 32000)

    Returns:
    - List of dictionaries, where each dictionary contains:
      - 'row_id': The timestamp in seconds (5, 10, 15, etc.)
      - And one key for each species_id with its confidence score
    """
    # Ensure correct sample rate for BirdNET
    # birdnet_sr = 48000
    # if sample_rate != birdnet_sr:
    #     audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=birdnet_sr)
    #     sample_rate = birdnet_sr

    analyzer = get_analyzer()
    recording = RecordingBuffer(
        analyzer,
        buffer=audio,
        rate=sample_rate,
        min_conf=1e-10
    )
    recording.analyze()

    # Calculate total audio duration and number of chunks
    num_seconds = int(len(audio) / sample_rate)
    num_5sec_chunks = (num_seconds + 4) // 5  # Ceiling division to include partial chunks

    # Define the time points for BirdNET 3-second chunks and our 5-second output chunks
    x1 = np.arange(1.5, num_seconds, 3)
    x2 = np.arange(2.5, num_seconds, 5)

    # Ensure x2 has at least one point (for very short audio)
    if len(x2) == 0 and num_seconds > 0:
        x2 = np.array([2.5])

    # Create row_ids for the final output
    row_ids = [n for n in range(5, num_5sec_chunks * 5 + 5, 5)]

    # If no chunks (extremely short audio), return empty list with proper structure
    if len(x2) == 0:
        # Load class labels
        class_labels = load_clef_labels()
        empty_result = []
        for i in range(max(1, num_5sec_chunks)):
            row_dict = {'row_id': 5 * (i + 1)}
            for species_id in class_labels:
                row_dict[species_id] = 0.0
            empty_result.append(row_dict)
        return empty_result

    # Load class labels
    class_labels = load_clef_labels()

    # Create a mapping from species ID to index position
    species_id_to_index = {species_id: idx for idx, species_id in enumerate(class_labels)}

    # Calculate required size for intermediate results array
    required_3sec_chunks = int(np.ceil(num_seconds / 3))

    # Create zero-filled array for all possible 3-second chunks
    bn_result = np.zeros((required_3sec_chunks, len(class_labels)))

    # Fill the result with BirdNet prediction
    for rec in recording.detections:
        species_id = get_species_id(rec['common_name'], rec['scientific_name'])
        if species_id is not None and species_id in species_id_to_index:
            time_idx = int(rec['start_time'] // 3)
            # Skip if time_idx is out of bounds
            if time_idx >= required_3sec_chunks:
                continue
            species_idx = species_id_to_index[species_id]
            bn_result[time_idx, species_idx] = rec['confidence']

    # Create x1 array matching the full bn_result size
    x1_full = np.arange(1.5, required_3sec_chunks * 3, 3)

    # Ensure we have matching sizes between x1_full and bn_result
    if len(x1_full) > bn_result.shape[0]:
        x1_full = x1_full[:bn_result.shape[0]]
    elif len(x1_full) < bn_result.shape[0]:
        bn_result = bn_result[:len(x1_full)]

    # Ensure we have at least one point for interpolation
    if len(x1_full) == 0:
        x1_full = np.array([1.5])
        bn_result = np.zeros((1, len(class_labels)))

    try:
        # Interpolate to 5-second chunks
        interpolated_result = CubicSpline(x1_full, bn_result, extrapolate=True)(x2)
    except Exception as e:
        print(f"Interpolation error: {e}")
        print(f"x1_full shape: {x1_full.shape}, bn_result shape: {bn_result.shape}, x2 shape: {x2.shape}")
        # Return zeros if interpolation fails
        interpolated_result = np.zeros((len(x2), len(class_labels)))

    # Ensure we have the right number of results
    if len(interpolated_result) < num_5sec_chunks:
        # Pad with zeros if we don't have enough results
        padding = np.zeros((num_5sec_chunks - len(interpolated_result), len(class_labels)))
        interpolated_result = np.vstack([interpolated_result, padding])
    elif len(interpolated_result) > num_5sec_chunks:
        # Truncate if we have too many results
        interpolated_result = interpolated_result[:num_5sec_chunks]

    # Ensure row_ids matches the interpolated result size
    row_ids = row_ids[:len(interpolated_result)]
    while len(row_ids) < len(interpolated_result):
        row_ids.append(row_ids[-1] + 5)

    # Convert to list of dictionaries
    result_list = []
    for i, row in enumerate(interpolated_result):
        # Create a dictionary for this row
        row_dict = {'row_id': row_ids[i]}

        # Add species confidence scores
        for j, species_id in enumerate(class_labels):
            row_dict[species_id] = float(row[j])  # Convert numpy float to Python float

        result_list.append(row_dict)

    return result_list
