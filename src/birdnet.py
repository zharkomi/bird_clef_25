import librosa
import numpy as np
import tensorflow as tf

# Global variables to store loaded model and labels
_interpreter = None
_labels = None
_species_df = None
_scientific_to_id = {}
_common_to_id = {}

TFILE = "bn/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"
LABELS_FILE = "bn/BirdNET_GLOBAL_6K_V2.4_Labels.txt"


# BirdNET model loading and prediction functions
def load_model():
    """Load the BirdNET model from the specified path."""
    global _interpreter
    if _interpreter is None:
        _interpreter = tf.lite.Interpreter(model_path=TFILE)
        _interpreter.allocate_tensors()
    return _interpreter


def predict(interpreter, audio_data, sample_rate=48000):
    """Make a prediction using the BirdNET model."""
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Ensure audio is correct shape
    if len(audio_data.shape) == 1:
        audio_data = np.expand_dims(audio_data, axis=0)

    # Check if model expects specific input shape
    input_shape = input_details[0]['shape']
    if input_shape[1] != audio_data.shape[1]:
        # Reshape audio to match model's expected input
        # Fix for librosa.util.fix_length error (it requires size parameter)
        audio_data = np.expand_dims(librosa.util.fix_length(audio_data[0], size=input_shape[1]), axis=0)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], audio_data.astype(np.float32))

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return output_data


def load_labels():
    """Load species labels from file."""
    global _labels
    if _labels is None:
        with open(LABELS_FILE, 'r', encoding='utf-8') as f:
            _labels = [line.strip() for line in f]
    return _labels


def load_species_data(csv_path):
    """
    Load species data from CSV and create mappings for scientific and common names.

    Parameters:
    - csv_path: Path to the CSV file with species information

    Returns:
    - True if successful, False otherwise
    """
    global _species_df, _scientific_to_id, _common_to_id

    try:
        import pandas as pd
        _species_df = pd.read_csv(csv_path)
        # Create a mapping from scientific name to species ID
        _scientific_to_id = dict(zip(_species_df['scientific_name'].str.lower(), _species_df['primary_label']))
        # Create a mapping from common name to species ID (for species that have common names)
        _common_to_id = dict(zip(_species_df['common_name'].str.lower(), _species_df['primary_label']))
        return True
    except Exception as e:
        print(f"Error loading species data: {e}")
        return False


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
                               species_csv_path, class_labels,
                               chunk_duration=5,
                               sample_rate=32000):
    """
    Analyze bird sounds in fixed, non-overlapping chunks and map to required class labels.
    Normalizes the prediction values so that each row sums to 1.

    Parameters:
    - audio: Audio signal as numpy array
    - model_path: Path to the BirdNET model
    - labels_file: Path to the file containing BirdNET species labels
    - species_csv_path: Path to the CSV file with species mapping information
    - class_labels: List of class labels required for prediction output
    - chunk_duration: Duration of each chunk in seconds (default: 5)
    - sample_rate: Sample rate of the audio

    Returns:
    - List of dictionaries, one for each chunk. Each dictionary contains:
      - 'chunk_id': The chunk number (starting from 0)
      - 'time_end': The end time of the chunk in seconds
      - And one key for each species_id in class_labels with its normalized probability value
    """
    # Load model and labels
    interpreter = load_model()
    labels = load_labels()

    # Load species data if not already loaded
    if _species_df is None:
        load_species_data(species_csv_path)

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
        chunk_result['row_id'] = chunk_end // sample_rate

        # Skip prediction if chunk is too short
        if len(chunk) < sample_rate:
            # Normalize: for all zeros, set each value to 1/n
            num_species = len(class_labels)
            for species_id in class_labels:
                chunk_result[species_id] = 1.0 / num_species

            results.append(chunk_result)
            continue

        # Normalize chunk (with safety check for silent segments)
        max_abs_val = np.max(np.abs(chunk))
        if max_abs_val > 0:
            chunk = chunk / max_abs_val

        # Make prediction
        prediction = predict(interpreter, chunk)

        # Map BirdNET predictions to our class labels
        for idx, prob in enumerate(prediction[0]):
            species_full = labels[idx]

            # Split into scientific and common names
            scientific_name, common_name = split_species_safely(species_full)

            # Find matching species ID
            species_id = None
            if scientific_name in _scientific_to_id:
                species_id = _scientific_to_id[scientific_name]
            elif common_name and common_name in _common_to_id:
                species_id = _common_to_id[common_name]

            # Update probability if species is in our class_labels
            if species_id is not None and species_id in class_labels:
                # Use the higher value if we already have a prediction for this species
                current_prob = chunk_result.get(species_id, 0.0)
                chunk_result[species_id] = max(current_prob, float(prob))

        # Normalize the row values to sum to 1
        species_values = [chunk_result[species_id] for species_id in class_labels]
        total_sum = sum(species_values)

        # If all values are zero, set each to 1/n
        if total_sum == 0:
            num_species = len(class_labels)
            for species_id in class_labels:
                chunk_result[species_id] = 1.0 / num_species
        else:
            # Normalize non-zero values to sum to 1
            for species_id in class_labels:
                chunk_result[species_id] = chunk_result[species_id] / total_sum

        # Add result for this chunk
        results.append(chunk_result)

    return results
