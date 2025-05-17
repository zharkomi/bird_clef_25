import os
import argparse
import pandas as pd
import time
import random
import numpy as np
import traceback
import signal
import gc
import sys
from tqdm import tqdm
from contextlib import contextmanager

from augmentation_core import (
    load_voice_data,
    safe_augment_file,
    create_augmentation_presets,
    add_custom_background_noise
)

from file_statistics import calculate_total_duration


class TimeoutException(Exception):
    """Exception raised when a function times out."""
    pass


@contextmanager
def time_limit(seconds):
    """
    Context manager that raises a TimeoutException if execution takes longer than specified seconds.
    Uses signal module which is more robust than threading-based timeouts.
    """
    def signal_handler(signum, frame):
        raise TimeoutException(f"Timed out after {seconds} seconds")
    
    # Set the timeout handler
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)


def process_single_species(species_id, train_dir, min_duration_seconds, background_dir, augmentations_per_file,
                          voice_data_path, min_output_duration, file_timeout=60):
    """
    Process a single species for augmentation without using thread pools.
    
    Args:
        species_id: ID of the species to process
        train_dir: Base training directory
        min_duration_seconds: Minimum duration required for the species
        background_dir: Directory with background noises
        augmentations_per_file: Number of augmentations per original file
        voice_data_path: Path to voice segments data file
        min_output_duration: Minimum duration for each output file
        file_timeout: Maximum time to spend on a single file
        
    Returns:
        Tuple of (species_id, success, message, new_duration)
    """
    try:
        # Initialize random seed
        random.seed(os.getpid() + int(time.time() * 1000) % 10000)
        np.random.seed(os.getpid() + int(time.time() * 1000) % 10000)
    
        # Load voice data
        try:
            voice_data = load_voice_data(voice_data_path)
        except Exception as e:
            print(f"Error loading voice data: {e}")
            print(traceback.format_exc())
            voice_data = {}
    
        species_dir = os.path.join(train_dir, species_id)
        print(f"Processing species {species_id} in directory {species_dir}")
    
        # Skip if species directory doesn't exist
        if not os.path.isdir(species_dir):
            print(f"Warning: Directory for species {species_id} not found at {species_dir}")
            return species_id, False, "Directory not found", 0
    
        # Calculate total duration
        total_duration = calculate_total_duration(species_dir)
        print(f"Species {species_id}: Total duration = {total_duration:.2f} seconds")
    
        # If duration is sufficient, skip to next species
        if total_duration >= min_duration_seconds:
            print(f"Species {species_id} has sufficient audio ({total_duration:.2f} seconds). Skipping augmentation.")
            return species_id, True, "Already sufficient", total_duration
    
        # Get all audio files for this species, excluding already augmented files
        audio_files = [f for f in os.listdir(species_dir) if f.endswith('.ogg') and "_aug_" not in f]
    
        if not audio_files:
            print(f"Warning: No original audio files found for species {species_id}")
            return species_id, False, "No original audio files found", 0
    
        # Calculate how much more audio is needed
        additional_duration_needed = min_duration_seconds - total_duration
        print(f"Need to generate {additional_duration_needed:.2f} more seconds for {species_id}")
    
        # Track progress
        additional_duration_generated = 0
        success_count = 0
        failure_count = 0
        skip_count = 0
    
        # Create augmentation presets without AddBackgroundNoise
        augmentation_presets = create_augmentation_presets(None)
    
        # Count existing augmentation files
        existing_aug_files = set([f for f in os.listdir(species_dir) if f.endswith('.ogg') and "_aug_" in f])
        print(f"Found {len(existing_aug_files)} existing augmentation files for {species_id}")
    
        # Check duration of existing augmentation files
        for aug_file in existing_aug_files:
            try:
                aug_file_path = os.path.join(species_dir, aug_file)
                import soundfile as sf
                info = sf.info(aug_file_path)
                existing_duration = info.duration
                additional_duration_generated += existing_duration
            except Exception as e:
                print(f"Error reading existing augmentation file {aug_file}: {e}")
                print(traceback.format_exc())
    
        print(f"Existing augmentation files contribute {additional_duration_generated:.2f} seconds for {species_id}")
        
        # Calculate how many more augmentations we need
        avg_file_duration = total_duration / len(audio_files) if len(audio_files) > 0 else 1.0
        files_multiplier = int(np.ceil((additional_duration_needed - additional_duration_generated) / 
                              (avg_file_duration * augmentations_per_file)))
        
        # Create a list of files to process, potentially using each file multiple times
        files_to_process = []
        for _ in range(files_multiplier):
            files_to_process.extend(audio_files)
        
        # Shuffle for better diversity
        # random.shuffle(files_to_process)

        base_aug_idx = len(existing_aug_files)

        files_to_process.sort()
        # Process each file sequentially
        for file_idx, file_name in enumerate(files_to_process):
            file_path = os.path.join(species_dir, file_name)
            base_file_name = os.path.splitext(file_name)[0]
            
            print(f"Processing file {file_idx+1}/{len(files_to_process)}: {file_name}")
            
            # Check if we have enough audio now
            if additional_duration_generated >= additional_duration_needed:
                print(f"Reached target duration for {species_id}. Stopping augmentation.")
                break
            
            # Create multiple augmentations for this file
            for aug_idx in range(augmentations_per_file):
                # Force garbage collection between iterations to prevent memory leaks
                gc.collect()
                
                # Check if we have enough audio now
                if additional_duration_generated >= additional_duration_needed:
                    print(f"Reached target duration for {species_id}. Stopping augmentation.")
                    break
                
                # Calculate augmentation file index
                aug_file_idx = base_aug_idx + 1
                aug_file_name = f"{base_file_name}_aug_{aug_file_idx}.ogg"
                aug_file_path = os.path.join(species_dir, aug_file_name)
                
                # Skip if this augmentation already exists
                if aug_file_name in existing_aug_files or os.path.exists(aug_file_path):
                    skip_count += 1
                    print(f"Skipped creating {aug_file_name} as it already exists")
                    continue
                
                print(f"Creating augmentation #{aug_idx+1} for {file_name} -> {aug_file_name}")
                
                # Process the augmentation
                try:
                    with time_limit(file_timeout):
                        result = safe_augment_file(
                            file_path=file_path,
                            species_id=species_id,
                            voice_data=voice_data,
                            augmentation_presets=augmentation_presets,
                            min_output_duration=min_output_duration,
                            background_dir=background_dir,
                            aug_file_idx=aug_file_idx
                        )
                    
                    # Update counts based on result
                    if result['status'] == 'success':
                        success_count += 1
                        base_aug_idx += 1  # Increment the base index only on success
                        additional_duration_generated += result.get('duration', 0)
                        print(f"Successfully created {aug_file_name} with duration {result.get('duration', 0):.2f}s")
                    elif result['status'] == 'skipped':
                        skip_count += 1
                        print(f"Skipped {aug_file_name}: {result.get('reason', 'unknown reason')}")
                    else:
                        failure_count += 1
                        print(f"Failed to create {aug_file_name}: {result.get('error', 'unknown error')}")
                
                except TimeoutException as e:
                    # Handle timeout specifically 
                    failure_count += 1
                    print(f"Timeout processing {aug_file_name}: {str(e)}")
                    
                    # Force garbage collection after timeout
                    gc.collect()
                
                except Exception as e:
                    # Handle other exceptions
                    failure_count += 1
                    print(f"Error processing {aug_file_name}: {str(e)}")
            
            # Force garbage collection after processing file
            gc.collect()
        
        # Calculate new total duration
        new_total_duration = calculate_total_duration(species_dir)
        print(f"Species {species_id}: New total duration = {new_total_duration:.2f} seconds")
        print(f"Summary for {species_id}: {success_count} successful, {failure_count} failed, {skip_count} skipped")
    
        return species_id, True, f"Generated {success_count} successful augmentations", new_total_duration
    except Exception as e:
        print(f"Unexpected error processing species {species_id}: {str(e)}")
        traceback.print_exc()
        return species_id, False, f"Error: {str(e)}", 0


def create_augmentations(species_list=None,
                         train_dir="/home/mikhail/prj/bc_25_data/train_audio",
                         min_duration_seconds=60.0,
                         background_dir=None,
                         augmentations_per_file=3,
                         voice_data_path="/home/mikhail/prj/bird_clef_25/data/train_voice_data.pkl",
                         stats_csv=None,
                         min_output_duration=3.0,
                         file_timeout=60):
    """
    Create augmentations for species by spawning separate processes.

    Args:
        species_list: List of species IDs to process. If None, use all directories in train_dir.
        train_dir: Directory containing audio files organized by species
        min_duration_seconds: Minimum total duration required for each species (in seconds)
        background_dir: Optional directory containing background noise files
        augmentations_per_file: Number of different augmentations to create per original file
        voice_data_path: Path to pickle file with voice segments
        stats_csv: Path to CSV with species statistics. If provided, use it to filter species.
        min_output_duration: Minimum duration in seconds for each augmented file
        file_timeout: Maximum time (seconds) to spend on a single file before timing out
    """
    try:
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
    
        # Launch each species processing as a separate process using os.system
        print(f"Starting augmentation for {len(species_list)} species using separate processes")
        
        for species_id in tqdm(species_list, desc="Launching species processes"):
            # Construct command to run this script for a single species
            cmd = f"python {__file__} --process_species {species_id} --train_dir \"{train_dir}\" --min_duration {min_duration_seconds} "
            
            if background_dir:
                cmd += f"--background_dir \"{background_dir}\" "
                
            cmd += f"--augmentations_per_file {augmentations_per_file} --voice_data_path \"{voice_data_path}\" "
            cmd += f"--min_output_duration {min_output_duration} --file_timeout {file_timeout}"
            
            print(f"Launching process for species {species_id}")
            print(f"Command: {cmd}")
            
            # Launch the process and wait for it to complete
            exit_code = os.system(cmd)
            
            if exit_code != 0:
                print(f"Warning: Process for species {species_id} exited with code {exit_code}")
        
        print("\n--- Summary of Augmentation Process ---")
        print(f"Launched processes for {len(species_list)} species.")
        print("\nTo verify results, run statistics.py again to generate updated statistics.")
        
        return True
    except Exception as e:
        print(f"Critical error in augmentation process: {str(e)}")
        traceback.print_exc()
        return False


def main():
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Robust audio augmentation for BirdCLEF dataset with process-level parallelism")
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
        parser.add_argument("--file_timeout", type=int, default=60,
                            help="Maximum time (seconds) to spend on a single file before timing out")
                            
        # Add special argument for processing a single species
        parser.add_argument("--process_species", type=str, default=None,
                            help="Process only this specific species ID (used for child processes)")
        args = parser.parse_args()
        
        # If process_species is specified, process only that species and exit
        if args.process_species:
            print(f"Child process started for species: {args.process_species}")
            result = process_single_species(
                species_id=args.process_species,
                train_dir=args.train_dir,
                min_duration_seconds=args.min_duration,
                background_dir=args.background_dir,
                augmentations_per_file=args.augmentations_per_file,
                voice_data_path=args.voice_data_path,
                min_output_duration=args.min_output_duration,
                file_timeout=args.file_timeout
            )
            
            # Print the result for this species
            species_id, success, message, new_duration = result
            if success:
                print(f"Successfully processed {species_id}: {message} (New duration: {new_duration:.2f}s)")
                return 0  # Success exit code
            else:
                print(f"Failed to process {species_id}: {message}")
                return 1  # Error exit code
        
        # This is the main process
        # Check if statistics CSV exists
        if not os.path.exists(args.stats_csv):
            print(f"Statistics file {args.stats_csv} not found. Please run statistics.py first.")
            return 1
    
        # Load existing statistics
        stats_df = pd.read_csv(args.stats_csv)
    
        # Print summary statistics
        print(f"Total number of species: {len(stats_df)}")
        print(f"Mean total duration across species: {stats_df['total_duration'].mean():.2f} seconds")
        print(f"Median total duration across species: {stats_df['total_duration'].median():.2f} seconds")
    
        # Get species that need augmentation (less than minimum duration)
        species_to_augment = stats_df[stats_df['total_duration'] < args.min_duration]['species_id'].tolist()
        print(f"Found {len(species_to_augment)} species that need augmentation")
    
        # Process all species by spawning separate processes
        success = create_augmentations(
            species_list=species_to_augment,
            train_dir=args.train_dir,
            min_duration_seconds=args.min_duration,
            background_dir=args.background_dir,
            augmentations_per_file=args.augmentations_per_file,
            voice_data_path=args.voice_data_path,
            stats_csv=args.stats_csv,
            min_output_duration=args.min_output_duration,
            file_timeout=args.file_timeout
        )
    
        return 0 if success else 1
    
    except Exception as e:
        print(f"Critical error in main function: {str(e)}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
