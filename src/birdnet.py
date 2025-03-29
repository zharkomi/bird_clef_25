import librosa
import numpy as np
import tensorflow as tf

# Global variables to store loaded model and labels
_interpreter = None
_labels = None


# BirdNET model loading and prediction functions
def load_model(model_path):
    """Load the BirdNET model from the specified path."""
    global _interpreter
    if _interpreter is None:
        _interpreter = tf.lite.Interpreter(model_path=model_path)
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
        audio_data = librosa.util.fix_length(audio_data[0], input_shape[1])
        audio_data = np.expand_dims(audio_data, axis=0)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], audio_data.astype(np.float32))

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return output_data


def load_labels(labels_file):
    """Load species labels from file."""
    global _labels
    if _labels is None:
        with open(labels_file, 'r', encoding='utf-8') as f:
            _labels = [line.strip() for line in f]
    return _labels


def analyze_audio(audio_input, model_path, labels_file, confidence_threshold=0.5, sample_rate=None):
    """
    Analyze bird sounds in an audio file or audio signal.

    Parameters:
    - audio_input: Path to the audio file to analyze OR a tuple of (audio_signal, sample_rate)
    - model_path: Path to the BirdNET model
    - labels_file: Path to the file containing species labels
    - confidence_threshold: Minimum confidence score to include in results
    - sample_rate: Sample rate of the audio (only used if audio_input is a signal)

    Returns:
    - List of dictionaries containing species and confidence scores
    """
    # Load model and labels (uses cached versions if already loaded)
    interpreter = load_model(model_path)
    labels = load_labels(labels_file)

    audio = audio_input
    sr = sample_rate if sample_rate is not None else 48000

    # Ensure correct sample rate
    if sr != 48000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
        sr = 48000

    # Create sliding windows (3-second segments with 1.5-second overlap)
    window_size = 3 * sr
    hop_size = window_size // 2
    results = []

    for i in range(0, len(audio) - window_size + 1, hop_size):
        segment = audio[i:i + window_size]

        # Normalize segment (with safety check for silent segments)
        max_abs_val = np.max(np.abs(segment))
        if max_abs_val > 0:
            segment = segment / max_abs_val
        # If max_abs_val is 0, the segment is silent, so we leave it as zeros

        # Make prediction
        prediction = predict(interpreter, segment)

        # Get top predictions
        top_indices = np.argsort(prediction[0])[::-1]

        # Calculate timestamp
        start_time = i / sr
        end_time = (i + window_size) / sr
        timestamp = f"{start_time:.2f}-{end_time:.2f}"

        # Add results above threshold
        segment_results = []
        for idx in top_indices:
            confidence = prediction[0][idx]
            if confidence >= confidence_threshold:
                segment_results.append({
                    "species": labels[idx],
                    "confidence": float(confidence),
                    "timestamp": timestamp
                })

        if segment_results:
            results.extend(segment_results)

    # Sort results by confidence
    results.sort(key=lambda x: x["confidence"], reverse=True)

    return results