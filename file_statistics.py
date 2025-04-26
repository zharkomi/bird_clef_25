import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa
import matplotlib.pyplot as plt


def calculate_species_statistics(train_dir, output_csv="species_statistics.csv"):
    """
    Calculate statistics for each species and save to a CSV file.

    Args:
        train_dir: Directory containing audio files organized by species
        output_csv: Path to save the statistics CSV file

    Returns:
        DataFrame with species statistics
    """
    stats = []

    # Get all species directories
    species_list = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]

    for species_id in tqdm(species_list, desc="Calculating species statistics"):
        species_dir = os.path.join(train_dir, species_id)

        # Get all audio files for this species
        audio_files = [f for f in os.listdir(species_dir) if f.endswith('.ogg')]
        num_files = len(audio_files)

        if num_files == 0:
            print(f"Warning: No audio files found for species {species_id}")
            continue

        # Calculate total duration and durations of individual files
        total_duration = 0
        file_durations = []

        for file_name in audio_files:
            file_path = os.path.join(species_dir, file_name)
            try:
                # Get audio duration
                duration = librosa.get_duration(path=file_path)
                total_duration += duration
                file_durations.append(duration)
            except Exception as e:
                print(f"Error getting duration for {file_path}: {e}")

        # Calculate statistics
        avg_duration = total_duration / num_files if num_files > 0 else 0
        min_duration = min(file_durations) if file_durations else 0
        max_duration = max(file_durations) if file_durations else 0
        median_duration = np.median(file_durations) if file_durations else 0

        # Add to statistics list
        stats.append({
            'species_id': species_id,
            'num_files': num_files,
            'total_duration': total_duration,
            'avg_duration': avg_duration,
            'min_duration': min_duration,
            'max_duration': max_duration,
            'median_duration': median_duration
        })

    # Convert to DataFrame and sort by total duration
    stats_df = pd.DataFrame(stats)
    stats_df = stats_df.sort_values('total_duration')

    # Save to CSV
    stats_df.to_csv(output_csv, index=False)

    # Print summary statistics
    print(f"\nTotal number of species: {len(stats_df)}")
    print(f"Mean total duration across species: {stats_df['total_duration'].mean():.2f} seconds")
    print(f"Median total duration across species: {stats_df['total_duration'].median():.2f} seconds")

    return stats_df


def plot_species_duration_histogram(stats_df, output_path="species_duration_histogram.png"):
    """
    Plot a sorted histogram of species durations.

    Args:
        stats_df: DataFrame with species statistics
        output_path: Path to save the histogram image
    """
    plt.figure(figsize=(15, 8))

    # Sort species by total duration for the bar chart
    sorted_df = stats_df.sort_values('total_duration')

    # Plot bar chart
    plt.bar(range(len(sorted_df)), sorted_df['total_duration'], color='skyblue')

    # Add a horizontal line at 60 seconds (1 minute)
    plt.axhline(y=60, color='red', linestyle='--', label='1 minute threshold')

    # Add labels and title
    plt.xlabel('Species (sorted by duration)')
    plt.ylabel('Total Duration (seconds)')
    plt.title('Total Audio Duration by Species')
    plt.legend()

    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Histogram saved to {output_path}")


def calculate_total_duration(species_dir):
    """
    Calculate the total duration of all audio files in a species directory.

    Args:
        species_dir: Directory containing audio files for a species

    Returns:
        Total duration in seconds
    """
    total_duration = 0
    audio_files = [f for f in os.listdir(species_dir) if f.endswith('.ogg')]

    for file_name in audio_files:
        file_path = os.path.join(species_dir, file_name)
        try:
            # Get audio duration
            duration = librosa.get_duration(path=file_path)
            total_duration += duration
        except Exception as e:
            print(f"Error getting duration for {file_path}: {e}")

    return total_duration


if __name__ == "__main__":
    # Set paths
    TRAIN_DIR = "/home/mikhail/prj/bc_25_data/train_audio"
    STATS_CSV = "species_statistics.csv"

    # Calculate and save species statistics
    print(f"Calculating species statistics and saving to {STATS_CSV}...")
    stats_df = calculate_species_statistics(TRAIN_DIR, STATS_CSV)

    # Plot duration histogram
    plot_species_duration_histogram(stats_df)

    # Print species below threshold (e.g. 60 seconds)
    MIN_DURATION = 60.0  # 1 minute minimum
    species_to_augment = stats_df[stats_df['total_duration'] < MIN_DURATION]['species_id'].tolist()
    print(f"\nFound {len(species_to_augment)} species that need augmentation (below {MIN_DURATION} seconds)")

    # Output the species IDs that need augmentation
    if species_to_augment:
        print("Species requiring augmentation:")
        for species_id in species_to_augment:
            species_duration = stats_df[stats_df['species_id'] == species_id]['total_duration'].values[0]
            print(f"  - {species_id}: {species_duration:.2f} seconds")
