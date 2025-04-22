import os
import random
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa
import soundfile as sf
from audiomentations import (
    Compose, AddGaussianNoise, TimeStretch, PitchShift,
    Shift, Gain, RoomSimulator, AddBackgroundNoise
)

from src.audio import parse_file
from src.birdnet import get_analyzer, analyze_chunk, get_embedding


def load_voice_data(pkl_path="/home/mikhail/prj/bird_clef_25/data/train_voice_data.pkl"):
    """
    Load the pickle file containing human voice segments.

    Args:
        pkl_path: Path to the pickle file

    Returns:
        Dictionary mapping filenames to voice segments
    """
    with open(pkl_path, 'rb') as f:
        voice_data = pickle.load(f)

    print(f"Loaded voice data for {len(voice_data)} files")
    return voice_data


def remove_voice_segments(audio, sr, voice_segments):
    """
    Remove human voice segments from the audio by completely removing those parts
    rather than replacing with silence.

    Args:
        audio: Audio array
        sr: Sample rate
        voice_segments: List of dictionaries with 'start' and 'end' keys in seconds

    Returns:
        Audio with voice segments completely removed
    """
    # Sort voice segments by start time to ensure proper processing
    sorted_segments = sorted(voice_segments, key=lambda x: x['start'])

    # Initialize the result with an empty array
    result_audio = np.array([], dtype=audio.dtype)
    last_end = 0

    # Process each segment
    for segment in sorted_segments:
        start_time = segment['start']
        end_time = segment['end']

        # Convert time to samples
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        # Ensure indices are within bounds
        start_sample = max(0, min(start_sample, len(audio) - 1))
        end_sample = max(0, min(end_sample, len(audio) - 1))

        # Add the audio segment before this voice segment
        if start_sample > last_end:
            result_audio = np.concatenate([result_audio, audio[last_end:start_sample]])

        # Update the last ending position
        last_end = end_sample

    # Add the final segment after the last voice segment
    if last_end < len(audio):
        result_audio = np.concatenate([result_audio, audio[last_end:]])

    return result_audio


def remove_voice(audio, file_name, full_path, sr, voice_data):
    """
    Remove human voice from audio based on voice_data.

    Args:
        audio: Audio array
        file_name: Name of the file
        full_path: Full path of the file relative to training data root
        sr: Sample rate
        voice_data: Dictionary mapping filenames to voice segments

    Returns:
        Audio with voice segments removed
    """
    voice_segments = []
    # Check different possible keys for the voice data
    key_format = f'/kaggle/input/birdclef-2025/train_audio/{full_path}'
    if key_format in voice_data:
        voice_segments = voice_data[key_format]
    # Remove voice segments if present
    if voice_segments:
        print(f"Removing {len(voice_segments)} voice segments from {file_name}")
        audio = remove_voice_segments(audio, sr, voice_segments)
    return audio


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


def create_augmentation_pipeline(background_dir=None):
    """
    Create a pipeline of audio augmentations.

    Args:
        background_dir: Optional directory containing background noise files

    Returns:
        A Compose object with the augmentation pipeline
    """
    augmentations = [
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        Shift(min_fraction=-0.2, max_fraction=0.2, p=0.5),
        Gain(min_gain_in_db=-12, max_gain_in_db=12, p=0.5),
        RoomSimulator(p=0.3)
    ]

    # Add background noise if a directory with background sounds is provided
    if background_dir and os.path.exists(background_dir):
        augmentations.append(
            AddBackgroundNoise(
                sounds_path=background_dir,
                min_snr_in_db=3.0,
                max_snr_in_db=30.0,
                p=0.5
            )
        )

    return Compose(augmentations)


def create_diverse_augmentations(species_list=None,
                                 train_dir="/home/mikhail/prj/bc_25_data/train_audio",
                                 min_duration_seconds=60.0,
                                 background_dir=None,
                                 augmentations_per_file=3,
                                 voice_data_path="/home/mikhail/prj/bird_clef_25/data/train_voice_data.pkl"):
    """
    Create diverse augmentations for species with less than the minimum duration.
    This function creates multiple different augmentations for each file.

    Args:
        species_list: List of species IDs to process. If None, use all directories in train_dir.
        train_dir: Directory containing audio files organized by species
        min_duration_seconds: Minimum total duration required for each species (in seconds)
        background_dir: Optional directory containing background noise files
        augmentations_per_file: Number of different augmentations to create per original file
        voice_data_path: Path to pickle file with voice segments
    """
    # Load voice data
    voice_data = load_voice_data(voice_data_path)

    # If no species list is provided, use all directories in train_dir
    if species_list is None:
        species_list = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]

    print(f"Processing {len(species_list)} species with diverse augmentations...")

    # Define different augmentation presets
    augmentation_presets = [
        # Preset 1: Light augmentation
        Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.5),
            TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
            PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
            Gain(min_gain_db=-6, max_gain_db=6, p=0.5)
        ]),

        # Preset 2: Moderate augmentation
        Compose([
            AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.02, p=0.7),
            TimeStretch(min_rate=0.8, max_rate=1.2, p=0.7),
            PitchShift(min_semitones=-3, max_semitones=3, p=0.7),
            Shift(min_shift=-0.1, max_shift=0.1, p=0.5),
            Gain(min_gain_db=-9, max_gain_db=9, p=0.6)
        ]),

        # Preset 3: Heavy augmentation
        Compose([
            AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.03, p=0.8),
            TimeStretch(min_rate=0.7, max_rate=1.3, p=0.8),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.8),
            Shift(min_shift=-0.2, max_shift=0.2, p=0.7),
            Gain(min_gain_db=-12, max_gain_db=12, p=0.7),
            RoomSimulator(p=0.6)
        ])
    ]

    # Add background noise augmenter if directory provided
    if background_dir and os.path.exists(background_dir):
        background_augmenter = Compose([
            AddBackgroundNoise(
                sounds_path=background_dir,
                min_snr_db=3.0,
                max_snr_db=30.0,
                p=1.0
            )
        ])
        augmentation_presets.append(background_augmenter)

    # Process each species
    for species_id in tqdm(species_list, desc="Processing species"):
        species_dir = os.path.join(train_dir, species_id)

        # Skip if species directory doesn't exist
        if not os.path.isdir(species_dir):
            print(f"Warning: Directory for species {species_id} not found at {species_dir}")
            continue

        # Calculate total duration
        total_duration = calculate_total_duration(species_dir)
        print(f"Species {species_id}: Total duration = {total_duration:.2f} seconds")

        # If duration is sufficient, skip to next species
        if total_duration >= min_duration_seconds:
            print(f"Species {species_id} has sufficient audio ({total_duration:.2f} seconds). Skipping augmentation.")
            continue

        # Get all audio files for this species
        audio_files = [f for f in os.listdir(species_dir) if f.endswith('.ogg')]

        if not audio_files:
            print(f"Warning: No audio files found for species {species_id}")
            continue

        # Calculate how much more audio is needed
        additional_duration_needed = min_duration_seconds - total_duration
        print(f"Need to generate {additional_duration_needed:.2f} more seconds for {species_id}")

        # Keep augmenting until we reach the minimum duration
        additional_duration_generated = 0
        augmentation_count = 0

        # Create a queue of files to process
        files_to_process = []
        while len(files_to_process) * augmentations_per_file < (
                additional_duration_needed / (total_duration / len(audio_files))):
            files_to_process.extend(audio_files)

        # Shuffle the queue for better diversity
        random.shuffle(files_to_process)

        # Process each file in the queue
        for file_name in files_to_process:
            if additional_duration_generated >= additional_duration_needed:
                break

            file_path = os.path.join(species_dir, file_name)

            try:
                # Load the audio file
                sr, audio = parse_file(file_path)

                # Remove human voice before augmentation
                full_path = os.path.join(species_id, file_name)
                audio = remove_voice(audio, file_name, full_path, sr, voice_data)

                # Skip files that are too short after voice removal
                if len(audio) < sr * 0.5:  # Skip if less than 0.5 seconds
                    print(f"Skipping {file_name} - too short after voice removal")
                    continue

                # Create multiple augmentations for this file
                for aug_idx in range(augmentations_per_file):
                    if additional_duration_generated >= additional_duration_needed:
                        break

                    # Choose a random augmentation preset
                    augmenter = random.choice(augmentation_presets)

                    # Apply augmentations
                    augmented_audio = augmenter(samples=audio, sample_rate=sr)

                    # Save the augmented audio
                    augmentation_count += 1
                    aug_file_name = f"{os.path.splitext(file_name)[0]}_aug_{augmentation_count}.ogg"
                    aug_file_path = os.path.join(species_dir, aug_file_name)

                    # Save the file
                    sf.write(aug_file_path, augmented_audio, sr)

                    # Update the additional duration generated
                    aug_duration = len(augmented_audio) / sr
                    additional_duration_generated += aug_duration

                    print(f"Generated augmented file {aug_file_name} ({aug_duration:.2f} seconds)")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

        # Calculate new total duration
        new_total_duration = calculate_total_duration(species_dir)
        print(f"Species {species_id}: New total duration = {new_total_duration:.2f} seconds")


if __name__ == "__main__":
    # Set paths
    TRAIN_DIR = "/home/mikhail/prj/bc_25_data/train_audio"
    BACKGROUND_DIR = None  # Set this to a directory with background noise files if available
    VOICE_DATA_PATH = "/home/mikhail/prj/bird_clef_25/data/train_voice_data.pkl"

    # Get all species directories
    all_classes = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]

    # Use the advanced diverse augmentation approach with voice removal
    create_diverse_augmentations(
        species_list=all_classes,
        train_dir=TRAIN_DIR,
        min_duration_seconds=60.0,  # 1 minute minimum
        background_dir=BACKGROUND_DIR,
        augmentations_per_file=3,  # Create 3 different augmentations per original file
        voice_data_path=VOICE_DATA_PATH
    )
