import librosa
import numpy as np
import tensorflow as tf


# BirdNET model loading and prediction functions
def load_model(model_path):
    """Load the BirdNET model from the specified path."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


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
    with open(labels_file, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f]
    return labels


def analyze_audio(audio_file, model_path, labels_file, confidence_threshold=0.5):
    """
    Analyze bird sounds in an audio file.

    Parameters:
    - audio_file: Path to the audio file to analyze
    - model_path: Path to the BirdNET model
    - labels_file: Path to the file containing species labels
    - confidence_threshold: Minimum confidence score to include in results

    Returns:
    - List of dictionaries containing species and confidence scores
    """
    # Load model and labels
    interpreter = load_model(model_path)
    labels = load_labels(labels_file)

    # Load and preprocess audio
    audio, sr = librosa.load(audio_file, sr=48000, mono=True)

    # Create sliding windows (3-second segments with 1.5-second overlap)
    window_size = 3 * sr
    hop_size = window_size // 2
    timestamps = []
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
