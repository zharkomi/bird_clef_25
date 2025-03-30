def merge_consecutive_segments(predictions):
    """
    Merge consecutive segments of the same species from BirdNET predictions.

    Args:
        predictions (list): List of prediction objects from analyze_audio, each containing
                           species, confidence, and timestamp.

    Returns:
        list: Merged predictions with:
              - Start time from first segment
              - End time from last segment
              - Max confidence from all merged segments
              - Other fields preserved
    """
    if not predictions:
        return []

    # Sort predictions by species and start time
    sorted_predictions = sorted(predictions, key=lambda x: (
        x["species"],
        float(x["timestamp"].split("-")[0])
    ))

    merged_predictions = []
    current_group = None

    for pred in sorted_predictions:
        # Extract species and time information
        species = pred["species"]
        start_time, end_time = map(float, pred["timestamp"].split("-"))
        confidence = pred["confidence"]

        # If this is a new group or different species
        if (current_group is None or
                current_group["species"] != species or
                start_time > current_group["end_time"]):

            # Save the previous group if it exists
            if current_group is not None:
                merged_predictions.append({
                    "species": current_group["species"],
                    "confidence": current_group["max_confidence"],
                    "timestamp": f"{current_group['start_time']:.2f}-{current_group['end_time']:.2f}"
                })

            # Start a new group
            current_group = {
                "species": species,
                "start_time": start_time,
                "end_time": end_time,
                "max_confidence": confidence
            }
        else:
            # Extend the current group and update max confidence
            current_group["end_time"] = max(current_group["end_time"], end_time)
            current_group["max_confidence"] = max(current_group["max_confidence"], confidence)

    # Add the last group
    if current_group is not None:
        merged_predictions.append({
            "species": current_group["species"],
            "confidence": current_group["max_confidence"],
            "timestamp": f"{current_group['start_time']:.2f}-{current_group['end_time']:.2f}"
        })

    # If input data had probability, preserve that field structure
    if "probability" in predictions[0]:
        for pred in merged_predictions:
            # Set probability to same value as confidence for now
            # (You may need to adjust this based on how calculate_probability works)
            pred["probability"] = pred["confidence"]

    return merged_predictions


# Example usage
if __name__ == "__main__":
    # Example data
    sample_predictions = [
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
        },
        {
            "species": "Myiarchus tuberculifer_Dusky-capped Flycatcher",
            "confidence": 1.85,
            "timestamp": "4.50-7.50"
        },
        {
            "species": "Myiarchus tuberculifer_Dusky-capped Flycatcher",
            "confidence": 2.15,
            "timestamp": "7.50-10.50"
        }
    ]

    merged = merge_consecutive_segments(sample_predictions)

    import json

    print(json.dumps(merged, indent=4))