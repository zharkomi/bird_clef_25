import os
import pickle

import numpy as np
import tensorflow as tf

# Global configuration variables
MODEL_PATH = None
LABEL_ENCODER_PATH = None

# Global cache for model interpreters and label encoders
_MODEL_CACHE = {}
_LABEL_ENCODER_CACHE = {}


def _load_tflite_interpreter(model_path=None):
    """Load and cache TFLite interpreter"""
    # Use global path if none provided
    if model_path is None:
        if MODEL_PATH is None:
            raise ValueError("No model path specified. Call set_model_paths() first or provide path explicitly.")
        model_path = MODEL_PATH

    if model_path not in _MODEL_CACHE:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        # Cache interpreter and its details
        _MODEL_CACHE[model_path] = {
            'interpreter': interpreter,
            'input_details': interpreter.get_input_details(),
            'output_details': interpreter.get_output_details()
        }

    return _MODEL_CACHE[model_path]


def _load_label_encoder(label_encoder_path=None):
    """Load and cache label encoder"""
    # Use global path if none provided
    if label_encoder_path is None:
        if LABEL_ENCODER_PATH is None:
            raise ValueError(
                "No label encoder path specified. Call set_model_paths() first or provide path explicitly.")
        label_encoder_path = LABEL_ENCODER_PATH

    if label_encoder_path not in _LABEL_ENCODER_CACHE:
        if not os.path.exists(label_encoder_path):
            raise FileNotFoundError(f"Label encoder file not found: {label_encoder_path}")

        with open(label_encoder_path, 'rb') as f:
            _LABEL_ENCODER_CACHE[label_encoder_path] = pickle.load(f)

    return _LABEL_ENCODER_CACHE[label_encoder_path]


def calculate_birdnet_confidence(probabilities):
    """
    Calculate confidence scores similar to BirdNET.

    Args:
        probabilities: Model softmax probability outputs as numpy array

    Returns:
        Dictionary mapping species to confidence scores
    """
    # Apply sigmoid-like scaling to emphasize high confidence predictions
    # and de-emphasize low confidence predictions
    max_prob = np.max(probabilities)

    # Scale confidence values - using power transform
    confidence_values = {}
    for i, prob in enumerate(probabilities):
        # Skip zero or very low probabilities
        if prob < 0.01:
            continue

        # BirdNET-style scaling: power transform and percentage
        confidence = np.power(prob, 1.5) * 100
        confidence_values[i] = confidence

    return confidence_values


def predict_species_probabilities(embedding, min_confidence=0.0, max_results=None):
    """
    Predict species with BirdNET-like confidence scores from an audio embedding
    using the TFLite ensemble model.
    Uses cached model and label encoder for efficiency.

    Args:
        embedding: A numpy array of shape (1024,) containing the audio embedding
        min_confidence: Minimum confidence threshold (0-100) to include in results
        max_results: Maximum number of results to return (None for all)

    Returns:
        A dictionary mapping species names to their confidence scores, sorted by confidence
    """
    # Validate embedding shape
    if not isinstance(embedding, np.ndarray):
        try:
            embedding = np.array(embedding, dtype=np.float32)
        except:
            raise ValueError("Embedding must be convertible to a numpy array")

    if embedding.shape != (1024,):
        raise ValueError(f"Embedding must have shape (1024,), got {embedding.shape}")

    # Get cached interpreter and details
    model_data = _load_tflite_interpreter(MODEL_PATH)
    interpreter = model_data['interpreter']
    input_details = model_data['input_details']
    output_details = model_data['output_details']

    # Prepare input tensor
    # TFLite models typically expect batch dimension, so reshape if needed
    input_shape = input_details[0]['shape']
    if len(input_shape) > 1 and input_shape[0] == 1:
        # Model expects batch dimension
        input_data = embedding.reshape(1, -1).astype(np.float32)
    else:
        input_data = embedding.astype(np.float32)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # If output has batch dimension, remove it
    if len(output_data.shape) > 1 and output_data.shape[0] == 1:
        probabilities = output_data[0]
    else:
        probabilities = output_data

    # Calculate confidence scores from probabilities
    confidence_dict = calculate_birdnet_confidence(probabilities)

    # Get cached label encoder
    label_encoder = _load_label_encoder(LABEL_ENCODER_PATH)

    # Create a dictionary mapping species names to confidence scores
    species_confidence = {}
    for idx, confidence in confidence_dict.items():
        # Apply minimum confidence threshold
        if confidence >= min_confidence:
            species = label_encoder.classes_[idx]
            species_confidence[species] = float(confidence)

    # Sort by confidence (descending)
    sorted_results = dict(sorted(species_confidence.items(), key=lambda x: x[1], reverse=True))

    # Limit results if requested
    if max_results is not None and max_results > 0:
        # Keep only the top max_results entries
        sorted_results = dict(list(sorted_results.items())[:max_results])

    return sorted_results
