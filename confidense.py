import pandas as pd
import numpy as np
import json
from scipy.special import expit  # This is the logistic sigmoid function

from src.utils import calculate_probability

# Sample data
predictions = [
    {
        "species": "Crypturellus soui_Little Tinamou",
        "confidence": 4.132401943206787,
        "timestamp": "0.00-3.00"
    },
    {
        "species": "Crypturellus soui_Little Tinamou",
        "confidence": 2.995702028274536,
        "timestamp": "3.00-6.00"
    },
    {
        "species": "Crypturellus soui_Little Tinamou",
        "confidence": 2.67470645904541,
        "timestamp": "1.50-4.50"
    }
]

# Since we don't have validation data, we'll use the direct sigmoid approach
probabilities = calculate_probability(predictions)

# Print individual results for inspection
for pred in probabilities:
    print(f"Species: {pred['species']}")
    print(f"Confidence (logit): {pred['confidence']:.4f}")
    print(f"Probability: {pred['probability']:.4f}")
    print("-" * 50)

# Print the resulting array as JSON
print("\nResulting JSON array:")
print(json.dumps(probabilities, indent=4))