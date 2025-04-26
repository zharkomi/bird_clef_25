import os
import random
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import soundfile as sf
import concurrent.futures
import argparse
import multiprocessing as mp
from functools import partial
import time

from audiomentations import (
    Compose, AddGaussianNoise, TimeStretch, PitchShift,
    Shift, Gain, RoomSimulator, AddBackgroundNoise
)

from src.audio import parse_file
from file_statistics import calculate_total_duration


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


def extend_short_audio(audio, sr, min_duration=3.0, background_dir=None):
    """
    Extend audio files shorter than min_duration by adding background noise
    or repeating the audio to reach the minimum duration.

    Args:
        audio: Audio array
        sr: Sample rate
        min_duration: Minimum duration in seconds
        background_dir: Directory containing background noise files

    Returns:
        Extended audio that meets the minimum duration
    """
    # Calculate current duration in seconds
    current_duration = len(audio) / sr

    # If duration is already sufficient, return the original audio
    if current_duration >= min_duration:
        return audio

    # Calculate how much audio to add in seconds
    additional_duration_needed = min_duration - current_duration
    print(f"Extending audio: current={current_duration:.2f}s, adding={additional_duration_needed:.2f}s")

    # If background directory is provided and exists, use background sounds
    if background_dir and os.path.exists(background_dir):
        # Get all background files
        bg_files = []
        for root, _, files in os.walk(background_dir):
            for file in files:
                if file.endswith(('.wav', '.ogg', '.mp3')):
                    bg_files.append(os.path.join(root, file))

        if bg_files:
            # Randomly select a background file
            bg_file = random.choice(bg_files)
            try:
                # Load background audio
                bg_sr, bg_audio = parse_file(bg_file)

                # Resample background audio if needed
                if bg_sr != sr:
                    from librosa import resample
                    bg_audio = resample(y=bg_audio, orig_sr=bg_sr, target_sr=sr)

                # Ensure background audio is long enough
                if len(bg_audio) < int(additional_duration_needed * sr):
                    # Repeat background audio if necessary
                    repetitions = int(np.ceil((additional_duration_needed * sr) / len(bg_audio)))
                    bg_audio = np.tile(bg_audio, repetitions)

                # Trim background audio to required length
                bg_audio = bg_audio[:int(additional_duration_needed * sr)]

                # Apply volume reduction to background audio
                bg_volume_factor = 0.3  # 30% of original volume
                bg_audio = bg_audio * bg_volume_factor

                # Concatenate original audio with background
                extended_audio = np.concatenate([audio, bg_audio])
                return extended_audio

            except Exception as e:
                print(f"Error using background audio: {e}")
                # Fall back to repeating original audio if background fails
                pass

    # If no background directory or an error occurred, repeat the original audio
    if len(audio) == 0:
        # If audio is empty (e.g., after voice removal), generate silence
        extended_audio = np.zeros(int(min_duration * sr), dtype=np.float32)
    else:
        # Calculate how many times to repeat the audio
        repetitions = int(np.ceil(min_duration / current_duration))
        extended_audio = np.tile(audio, repetitions)

        # Trim to required length
        extended_audio = extended_audio[:int(min_duration * sr)]

    return extended_audio


def create_augmentation_presets(background_dir=None):
    """
    Create multiple augmentation presets.

    Args:
        background_dir: Optional directory containing background noise files

    Returns:
        List of Compose objects with different augmentation settings
    """
    presets = [
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
        presets.append(background_augmenter)

    return presets


def process_single_species(species_id, train_dir, min_duration_seconds, background_dir, augmentations_per_file,
                           voice_data_path, min_output_duration):
    """
    Process a single species for augmentation.
    Each process is completely independent and does not share state with other processes.

    Args:
        species_id: ID of the species to process
        train_dir: Base training directory
        min_duration_seconds: Minimum duration required for the species
        background_dir: Directory with background noises
        augmentations_per_file: Number of augmentations per original file
        voice_data_path: Path to voice segments data file
        min_output_duration: Minimum duration for each output file

    Returns:
        Tuple of (species_id, success, message, new_duration)
    """
    # Initialize process-specific random seed using current time and PID
    # This ensures each process has a unique random seed
    random.seed(os.getpid() + int(time.time() * 1000) % 10000)
    np.random.seed(os.getpid() + int(time.time() * 1000) % 10000)

    # Use process-local logging to avoid conflicts between processes
    def log_message(message):
        # Adding process ID ensures we can trace which process generated which message
        print(f"[Process {os.getpid()}, Species {species_id}] {message}")

    # Load voice data in this process - each process loads its own copy
    # This eliminates shared state between processes
    try:
        with open(voice_data_path, 'rb') as f:
            voice_data = pickle.load(f)
        log_message(f"Loaded voice data with {len(voice_data)} entries")
    except Exception as e:
        log_message(f"Error loading voice data: {e}")
        voice_data = {}

    species_dir = os.path.join(train_dir, species_id)

    # Skip if species directory doesn't exist
    if not os.path.isdir(species_dir):
        log_message(f"Warning: Directory for species {species_id} not found at {species_dir}")
        return species_id, False, "Directory not found", 0

    # Calculate total duration
    total_duration = calculate_total_duration(species_dir)
    log_message(f"Species {species_id}: Total duration = {total_duration:.2f} seconds")

    # If duration is sufficient, skip to next species
    if total_duration >= min_duration_seconds:
        log_message(f"Species {species_id} has sufficient audio ({total_duration:.2f} seconds). Skipping augmentation.")
        return species_id, True, "Already sufficient", total_duration

    # Get all audio files for this species
    audio_files = [f for f in os.listdir(species_dir) if f.endswith('.ogg')]

    if not audio_files:
        log_message(f"Warning: No audio files found for species {species_id}")
        return species_id, False, "No audio files found", 0

    # Calculate how much more audio is needed
    additional_duration_needed = min_duration_seconds - total_duration
    log_message(f"Need to generate {additional_duration_needed:.2f} more seconds for {species_id}")

    # Keep augmenting until we reach the minimum duration
    additional_duration_generated = 0
    augmentation_count = 0

    # Create augmentation presets
    augmentation_presets = create_augmentation_presets(background_dir)

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

            # Check if audio is too short after voice removal
            current_duration = len(audio) / sr
            if current_duration < 0.5:  # If less than 0.5 seconds, skip
                log_message(f"Skipping {file_name} - too short after voice removal (only {current_duration:.2f}s)")
                continue

            # If audio is less than min_output_duration, extend it
            if current_duration < min_output_duration:
                log_message(
                    f"Extending audio file {file_name} from {current_duration:.2f}s to {min_output_duration:.2f}s")
                audio = extend_short_audio(audio, sr, min_output_duration, background_dir)

            # Create multiple augmentations for this file
            for aug_idx in range(augmentations_per_file):
                if additional_duration_generated >= additional_duration_needed:
                    break

                # Choose a random augmentation preset
                augmenter = random.choice(augmentation_presets)

                # Apply augmentations
                augmented_audio = augmenter(samples=audio, sample_rate=sr)

                # Check if augmented audio is too short and extend if needed
                aug_duration = len(augmented_audio) / sr
                if aug_duration < min_output_duration:
                    log_message(f"Extending augmented audio from {aug_duration:.2f}s to {min_output_duration:.2f}s")
                    augmented_audio = extend_short_audio(augmented_audio, sr, min_output_duration, background_dir)
                    aug_duration = len(augmented_audio) / sr

                # Save the augmented audio
                augmentation_count += 1
                aug_file_name = f"{os.path.splitext(file_name)[0]}_aug_{augmentation_count}.ogg"
                aug_file_path = os.path.join(species_dir, aug_file_name)

                # Save the file
                sf.write(aug_file_path, augmented_audio, sr)

                # Update the additional duration generated
                additional_duration_generated += aug_duration

                log_message(f"Generated augmented file {aug_file_name} ({aug_duration:.2f} seconds)")

        except Exception as e:
            log_message(f"Error processing {file_path}: {e}")
            continue

    # Calculate new total duration
    new_total_duration = calculate_total_duration(species_dir)
    log_message(f"Species {species_id}: New total duration = {new_total_duration:.2f} seconds")

    return species_id, True, f"Generated {augmentation_count} augmentations", new_total_duration


def create_parallel_augmentations(species_list=None,
                                  train_dir="/home/mikhail/prj/bc_25_data/train_audio",
                                  min_duration_seconds=60.0,
                                  background_dir=None,
                                  augmentations_per_file=3,
                                  voice_data_path="/home/mikhail/prj/bird_clef_25/data/train_voice_data.pkl",
                                  stats_csv=None,
                                  min_output_duration=3.0,
                                  num_processes=4):
    """
    Create augmentations for species in parallel using multiple processes.

    Args:
        species_list: List of species IDs to process. If None, use all directories in train_dir.
        train_dir: Directory containing audio files organized by species
        min_duration_seconds: Minimum total duration required for each species (in seconds)
        background_dir: Optional directory containing background noise files
        augmentations_per_file: Number of different augmentations to create per original file
        voice_data_path: Path to pickle file with voice segments
        stats_csv: Path to CSV with species statistics. If provided, use it to filter species.
        min_output_duration: Minimum duration in seconds for each augmented file
        num_processes: Number of processes to use for parallel processing
    """
    # If stats_csv is provided, use it to determine which species need augmentation
    if stats_csv and os.path.exists(stats_csv):
        print(f"Using existing statistics from {stats_csv}")
        stats_df = pd.read_csv(stats_csv)

        # Filter species that need augmentation
        species_to_augment = stats_df[stats_df['total_duration'] < min_duration_seconds]['species_id'].tolist()

        if species_list is not None:
            # Intersect with provided species list if given
            species_to_augment = [s for s in species_to_augment if s in species_list]

        print(f"Found {len(species_to_augment)} species that need augmentation based on statistics")
        species_list = species_to_augment

    # If no species list is provided, use all directories in train_dir
    if species_list is None:
        species_list = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]

    print(f"Processing {len(species_list)} species with parallel augmentations using {num_processes} processes...")

    # Process species in parallel with complete independence
    try:
        # Create process pool
        with mp.Pool(processes=num_processes) as pool:
            # Create a partial function with all the fixed parameters
            process_func = partial(
                process_single_species,
                train_dir=train_dir,
                min_duration_seconds=min_duration_seconds,
                background_dir=background_dir,
                augmentations_per_file=augmentations_per_file,
                voice_data_path=voice_data_path,
                min_output_duration=min_output_duration
            )

            # Map the function to the species list and collect results directly
            # Using pool.map instead of imap_unordered to ensure we get a direct list result
            results = list(tqdm(
                pool.map(process_func, species_list),
                total=len(species_list),
                desc="Processing species"
            ))
    except Exception as e:
        print(f"Error during parallel processing: {e}")
        results = []

    # Print summary
    print("\n--- Summary of Augmentation Process ---")
    success_count = sum(1 for _, success, _, _ in results if success)
    print(f"Successfully processed {success_count} out of {len(species_list)} species.")

    for species_id, success, message, new_duration in results:
        if success:
            print(f"- {species_id}: {message} (New duration: {new_duration:.2f}s)")
        else:
            print(f"- {species_id}: FAILED - {message}")

    print("\nTo verify results, run statistics.py again to generate updated statistics.")
    return results


if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for better compatibility
    # This helps avoid issues with fork() on some platforms, especially with audio libraries
    # 'spawn' starts a fresh Python interpreter process, which ensures complete isolation
    # between processes - critical for independence
    mp.set_start_method('spawn', force=True)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Parallel audio augmentation for BirdCLEF dataset")
    parser.add_argument("--train_dir", type=str, default="/home/mikhail/prj/bc_25_data/train_audio",
                        help="Directory containing audio files organized by species")
    parser.add_argument("--background_dir", type=str, default="/home/mikhail/prj/bc_25_data/train_soundscapes",
                        help="Directory containing background noise files")
    parser.add_argument("--voice_data_path", type=str,
                        default="/home/mikhail/prj/bird_clef_25/data/train_voice_data.pkl",
                        help="Path to pickle file with voice segments")
    parser.add_argument("--stats_csv", type=str, default="species_statistics.csv",
                        help="Path to CSV with species statistics")
    parser.add_argument("--min_duration", type=float, default=2400.0,
                        help="Minimum total duration required for each species (in seconds)")
    parser.add_argument("--min_output_duration", type=float, default=3.0,
                        help="Minimum duration for each output file (in seconds)")
    parser.add_argument("--augmentations_per_file", type=int, default=3,
                        help="Number of different augmentations to create per original file")
    parser.add_argument("--processes", type=int, default=20,
                        help="Number of processes to use for parallel processing")
    args = parser.parse_args()

    # Check if statistics CSV exists
    if not os.path.exists(args.stats_csv):
        print(f"Statistics file {args.stats_csv} not found. Please run statistics.py first.")
        exit(1)

    # Load existing statistics
    stats_df = pd.read_csv(args.stats_csv)

    # Print summary statistics
    print(f"Total number of species: {len(stats_df)}")
    print(f"Mean total duration across species: {stats_df['total_duration'].mean():.2f} seconds")
    print(f"Median total duration across species: {stats_df['total_duration'].median():.2f} seconds")

    # Get species that need augmentation (less than minimum duration)
    species_to_augment = stats_df[stats_df['total_duration'] < args.min_duration]['species_id'].tolist()
    print(f"Found {len(species_to_augment)} species that need augmentation")

    # Process all species at once in parallel
    results = create_parallel_augmentations(
        species_list=species_to_augment,
        train_dir=args.train_dir,
        min_duration_seconds=args.min_duration,
        background_dir=args.background_dir,
        augmentations_per_file=args.augmentations_per_file,
        voice_data_path=args.voice_data_path,
        stats_csv=args.stats_csv,
        min_output_duration=args.min_output_duration,
        num_processes=args.processes
    )

    # Final summary
    print("\n=== FINAL AUGMENTATION SUMMARY ===")
    success_count = sum(1 for _, success, _, _ in results if success)
    print(f"Successfully processed {success_count} out of {len(species_to_augment)} species.")
