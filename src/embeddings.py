import os
import pickle

import numpy as np
import tensorflow as tf

# Global configuration variables
EMB_MODEL_PATH = None
EMB_LABEL_ENCODER_PATH = None

# Global cache for model interpreters and label encoders
_MODEL_CACHE = {}
_LABEL_ENCODER_CACHE = {}


def _load_tflite_interpreter(model_path=None):
    """Load and cache TFLite interpreter"""
    # Use global path if none provided
    if model_path is None:
        if EMB_MODEL_PATH is None:
            raise ValueError("No model path specified. Call set_model_paths() first or provide path explicitly.")
        model_path = EMB_MODEL_PATH

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
        if EMB_LABEL_ENCODER_PATH is None:
            raise ValueError(
                "No label encoder path specified. Call set_model_paths() first or provide path explicitly.")
        label_encoder_path = EMB_LABEL_ENCODER_PATH

    if label_encoder_path not in _LABEL_ENCODER_CACHE:
        if not os.path.exists(label_encoder_path):
            raise FileNotFoundError(f"Label encoder file not found: {label_encoder_path}")

        with open(label_encoder_path, 'rb') as f:
            _LABEL_ENCODER_CACHE[label_encoder_path] = pickle.load(f)

    return _LABEL_ENCODER_CACHE[label_encoder_path]


def flat_sigmoid(x, sensitivity=-1.0):
    """
    Apply the same custom sigmoid function used by BirdNET.

    Args:
        x: Input array or value
        sensitivity: Sigmoid sensitivity parameter, default -1.0

    Returns:
        Transformed values after applying custom sigmoid
    """
    return 1.0 / (1.0 + np.exp(sensitivity * np.clip(x, -15, 15)))


def calculate_birdnet_confidence(logits):
    """
    Calculate confidence scores exactly like BirdNET does.

    Args:
        logits: Raw model output logits as numpy array

    Returns:
        Confidence scores after applying BirdNET's sigmoid transformation
    """
    # Apply BirdNET's custom sigmoid transformation
    return flat_sigmoid(logits, sensitivity=-1.0)


def predict_species_probabilities(embedding, min_confidence=0.1):
    """
    Predict species with BirdNET-like confidence scores from an audio embedding
    using the TFLite ensemble model.
    Uses cached model and label encoder for efficiency.

    Args:
        embedding: A numpy array of shape (1024,) containing the audio embedding
        min_confidence: Minimum confidence threshold (0-1) to include in results

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
    model_data = _load_tflite_interpreter(EMB_MODEL_PATH)
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

    # Get the output tensor (these are logits, not probabilities)
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # If output has batch dimension, remove it
    if len(output_data.shape) > 1 and output_data.shape[0] == 1:
        logits = output_data[0]
    else:
        logits = output_data

    # Calculate BirdNET confidence scores from logits
    confidences = calculate_birdnet_confidence(logits)

    # Get cached label encoder
    label_encoder = _load_label_encoder(EMB_LABEL_ENCODER_PATH)

    # Create a dictionary mapping species names to confidence scores
    species_confidence = {}
    for idx, confidence in enumerate(confidences):
        # Apply minimum confidence threshold
        if confidence >= min_confidence:
            species = label_encoder.classes_[idx]
            species_confidence[species] = float(confidence)

    return species_confidence
