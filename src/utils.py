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


def calculate_probability(predictions, validation_data=None):
    """
    Calculate probability for each BirdNET prediction using either:
    1. The sigmoid function (if no validation data is provided)
    2. A logistic regression model (if validation data is provided)

    Args:
        predictions: List of dictionaries containing BirdNET predictions
        validation_data: Optional DataFrame with columns 'logit_score' and 'is_correct'

    Returns:
        List of dictionaries with added probability field
    """
    if len(predictions) == 0:
        return predictions

    # Convert predictions to a DataFrame for easier processing
    df = pd.DataFrame(predictions)

    if validation_data is None:
        # Method 1: Use the sigmoid function directly
        # Default sensitivity value (usually = 1 in BirdNET)
        sensitivity = 1.0

        # Calculate probability using the sigmoid function
        df['probability'] = expit(df['confidence'] * sensitivity)
    else:
        # Method 2: Use logistic regression based on validation data
        from sklearn.linear_model import LogisticRegression

        # Train a logistic regression model on validation data
        X = validation_data[['logit_score']]
        y = validation_data['is_correct']

        model = LogisticRegression().fit(X, y)

        # Use the model to predict probabilities
        df['probability'] = model.predict_proba(df[['confidence']])[:, 1]

    # Convert back to list of dictionaries
    result = df.to_dict('records')

    return result
