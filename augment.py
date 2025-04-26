import os
import argparse
import pandas as pd
import time
import random
import numpy as np
import traceback
import signal
import gc
from functools import partial
from tqdm import tqdm
import concurrent.futures
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


def process_input_file(args):
    """
    Process a single input file for augmentation, generating multiple augmented versions.
    
    Args:
        args: Tuple containing (file_path, species_id, voice_data, other_params)
        
    Returns:
        Dictionary with augmentation results
    """
    try:
        file_path, species_id, voice_data, augmentation_presets, min_output_duration, background_dir, augmentations_per_file, file_timeout, base_aug_idx = args
        
        results = []
        success_count = 0
        failure_count = 0
        skip_count = 0
        total_duration = 0
        error_messages = []
        
        file_name = os.path.basename(file_path)
        base_file_name = os.path.splitext(file_name)[0]
        species_dir = os.path.dirname(file_path)
        
        process_id = os.getpid()
        print(f"[Process {process_id}] Starting to process file: {file_path}")
        
        # Check for existing augmentation files
        existing_aug_files = set([f for f in os.listdir(species_dir) if f.startswith(f"{base_file_name}_aug_") and f.endswith('.ogg')])
        
        # Verify input file
        try:
            # Use the time limit context manager for file loading
            with time_limit(file_timeout // 2):  # Half timeout for file loading
                from src.audio import parse_file
                sr, audio = parse_file(file_path)
                print(f"[Process {process_id}] Successfully loaded {file_name} - Duration: {len(audio)/sr:.2f}s, Sample Rate: {sr}Hz")
        except Exception as e:
            error_message = f"ERROR: Could not load input file {file_name}: {str(e)}"
            print(f"[Process {process_id}] {error_message}")
            error_messages.append(error_message)
            return {
                'file_name': file_name,
                'results': [],
                'success_count': 0,
                'failure_count': augmentations_per_file,
                'skip_count': 0,
                'total_duration': 0,
                'error_messages': [f"{error_message}\n{traceback.format_exc()}"]
            }
        
        # Create multiple augmentations per input file
        for aug_idx in range(augmentations_per_file):
            # Force garbage collection between iterations to prevent memory leaks
            gc.collect()
            
            aug_file_idx = base_aug_idx + aug_idx
            aug_file_name = f"{base_file_name}_aug_{aug_file_idx}.ogg"
            
            # Skip if this augmentation already exists
            if aug_file_name in existing_aug_files:
                skip_count += 1
                skip_result = {
                    'status': 'skipped',
                    'file_name': file_name,
                    'aug_file_name': aug_file_name,
                    'reason': 'Augmentation file already exists'
                }
                results.append(skip_result)
                print(f"[Process {process_id}] Skipped creating {aug_file_name} as it already exists")
                continue
            
            print(f"[Process {process_id}] Creating augmentation #{aug_idx+1} for {file_name} -> {aug_file_name}")
            
            # Process the augmentation with timeout
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
                
                results.append(result)
                
                # Update counts based on result
                if result['status'] == 'success':
                    success_count += 1
                    total_duration += result.get('duration', 0)
                    print(f"[Process {process_id}] Successfully created {aug_file_name} with duration {result.get('duration', 0):.2f}s")
                elif result['status'] == 'skipped':
                    skip_count += 1
                    print(f"[Process {process_id}] Skipped {aug_file_name}: {result.get('reason', 'unknown reason')}")
                else:
                    failure_count += 1
                    error_msg = f"Failed to create {aug_file_name}: {result.get('error', 'unknown error')}"
                    print(f"[Process {process_id}] {error_msg}")
                    error_messages.append(error_msg)
            
            except TimeoutException as e:
                # Handle timeout specifically 
                failure_count += 1
                error_msg = f"Timeout processing {aug_file_name}: {str(e)}"
                print(f"[Process {process_id}] {error_msg}")
                error_messages.append(error_msg)
                results.append({
                    'status': 'error',
                    'file_name': file_name,
                    'aug_file_name': aug_file_name,
                    'error': f"Timeout after {file_timeout} seconds"
                })
                
                # Force garbage collection after timeout
                gc.collect()
            
            except Exception as e:
                # Handle other exceptions
                failure_count += 1
                error_msg = f"Error processing {aug_file_name}: {str(e)}"
                print(f"[Process {process_id}] {error_msg}")
                error_messages.append(error_msg)
                results.append({
                    'status': 'error',
                    'file_name': file_name,
                    'aug_file_name': aug_file_name,
                    'error': str(e)
                })
        
        return {
            'file_name': file_name,
            'results': results,
            'success_count': success_count,
            'failure_count': failure_count,
            'skip_count': skip_count,
            'total_duration': total_duration,
            'error_messages': error_messages
        }
    except Exception as e:
        # Catch any unexpected exceptions at the file level
        error_message = f"Error processing file {os.path.basename(file_path)}: {str(e)}"
        print(f"[Process {os.getpid()}] {error_message}")
        print(traceback.format_exc())
        return {
            'file_name': os.path.basename(file_path),
            'results': [],
            'success_count': 0,
            'failure_count': augmentations_per_file,
            'skip_count': 0,
            'total_duration': 0,
            'error_messages': [f"{error_message}\n{traceback.format_exc()}"]
        }


def process_batch(batch, species_id, train_dir, voice_data, augmentation_presets, min_output_duration, 
                  background_dir, augmentations_per_file, file_timeout, base_aug_idx):
    """
    Process a batch of files with a new process pool for each batch to prevent resource exhaustion.
    
    Args:
        batch: List of file names to process
        species_id: ID of the species to process
        train_dir: Base training directory
        voice_data: Voice data dictionary
        augmentation_presets: List of augmentation presets
        min_output_duration: Minimum duration for output files
        background_dir: Directory with background noises
        augmentations_per_file: Number of augmentations per file
        file_timeout: Maximum time to spend on a single file
        base_aug_idx: Base index for augmentation files
        
    Returns:
        List of results for each file
    """
    species_dir = os.path.join(train_dir, species_id)
    
    # Create tasks for each file in the batch
    tasks = []
    for i, file_name in enumerate(batch):
        file_path = os.path.join(species_dir, file_name)
        tasks.append((
            file_path, 
            species_id, 
            voice_data, 
            augmentation_presets, 
            min_output_duration, 
            background_dir, 
            augmentations_per_file,
            file_timeout,
            base_aug_idx + i * augmentations_per_file
        ))
    
    # Process batch with a fresh process pool
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(len(tasks), 4)) as executor:
        # Submit all tasks
        futures = [executor.submit(process_input_file, task) for task in tasks]
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Exception processing batch: {e}")
                print(traceback.format_exc())
                # Add a placeholder result for the failed task
                results.append({
                    'file_name': 'unknown',
                    'results': [],
                    'success_count': 0,
                    'failure_count': augmentations_per_file,
                    'skip_count': 0,
                    'total_duration': 0,
                    'error_messages': [f"Process pool error: {str(e)}"]
                })
    
    return results


def process_species_with_batch_parallelism(species_id, train_dir, min_duration_seconds, background_dir, augmentations_per_file,
                                           voice_data_path, min_output_duration, num_processes, file_timeout=60, batch_size=5):
    """
    Process a single species for augmentation with batch-level parallelism.
    
    Args:
        species_id: ID of the species to process
        train_dir: Base training directory
        min_duration_seconds: Minimum duration required for the species
        background_dir: Directory with background noises
        augmentations_per_file: Number of augmentations per original file
        voice_data_path: Path to voice segments data file
        min_output_duration: Minimum duration for each output file
        num_processes: Maximum number of parallel processes
        file_timeout: Maximum time to spend on a single file
        batch_size: Number of files to process in each batch
        
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
    
        # Count duration of existing augmentation files
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
    
        # Create enough tasks to meet the duration requirement
        avg_file_duration = total_duration / len(audio_files) if len(audio_files) > 0 else 1.0
        files_multiplier = int(np.ceil((additional_duration_needed - additional_duration_generated) / 
                              (avg_file_duration * augmentations_per_file)))
        
        # Create a list of files to process, potentially using each file multiple times
        files_to_process = []
        for _ in range(files_multiplier):
            files_to_process.extend(audio_files)
        
        # Shuffle for better diversity
        random.shuffle(files_to_process)
        
        # Organize files into batches
        batches = [files_to_process[i:i+batch_size] for i in range(0, len(files_to_process), batch_size)]
        print(f"Processing {species_id} in {len(batches)} batches of up to {batch_size} files each")
        
        base_aug_idx = len(existing_aug_files) + 1
        batch_results = []
        
        # Process batches sequentially, but files within batches in parallel
        for batch_idx, batch in enumerate(batches):
            print(f"Processing batch {batch_idx+1}/{len(batches)} for {species_id}")
            
            try:
                # Process this batch
                batch_result = process_batch(
                    batch=batch,
                    species_id=species_id,
                    train_dir=train_dir,
                    voice_data=voice_data,
                    augmentation_presets=augmentation_presets,
                    min_output_duration=min_output_duration,
                    background_dir=background_dir,
                    augmentations_per_file=augmentations_per_file,
                    file_timeout=file_timeout,
                    base_aug_idx=base_aug_idx
                )
                
                # Update base index for next batch
                base_aug_idx += len(batch) * augmentations_per_file
                
                # Process batch results
                for result in batch_result:
                    success_count += result['success_count']
                    failure_count += result['failure_count']
                    skip_count += result['skip_count']
                    additional_duration_generated += result['total_duration']
                    
                    # Print any error messages
                    if 'error_messages' in result and result['error_messages']:
                        print(f"Errors for {result['file_name']}:")
                        for error_msg in result['error_messages']:
                            print(f"  - {error_msg}")
                
                # Check if we have enough audio now
                if additional_duration_generated >= additional_duration_needed:
                    print(f"Reached target duration for {species_id}. Stopping augmentation.")
                    break
                
            except Exception as e:
                print(f"Error processing batch {batch_idx+1} for {species_id}: {e}")
                print(traceback.format_exc())
                # Continue with next batch instead of aborting everything
        
        # Force garbage collection
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
                         num_processes=4,
                         file_timeout=60,
                         batch_size=5):
    """
    Create augmentations for species with batch-level parallelism.

    Args:
        species_list: List of species IDs to process. If None, use all directories in train_dir.
        train_dir: Directory containing audio files organized by species
        min_duration_seconds: Minimum total duration required for each species (in seconds)
        background_dir: Optional directory containing background noise files
        augmentations_per_file: Number of different augmentations to create per original file
        voice_data_path: Path to pickle file with voice segments
        stats_csv: Path to CSV with species statistics. If provided, use it to filter species.
        min_output_duration: Minimum duration in seconds for each augmented file
        num_processes: Maximum number of parallel processes for file processing
        file_timeout: Maximum time (seconds) to spend on a single file before timing out
        batch_size: Number of files to process in each batch
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
    
        # If file_timeout is not specified, use a default value
        if file_timeout is None or file_timeout <= 0:
            file_timeout = 60
            print(f"Using default file processing timeout of {file_timeout} seconds")
        else:
            print(f"Using file processing timeout of {file_timeout} seconds")
    
        # Process species sequentially but files in parallel
        results = []
        for species_id in species_list:
            try:
                result = process_species_with_batch_parallelism(
                    species_id=species_id,
                    train_dir=train_dir,
                    min_duration_seconds=min_duration_seconds,
                    background_dir=background_dir,
                    augmentations_per_file=augmentations_per_file,
                    voice_data_path=voice_data_path,
                    min_output_duration=min_output_duration,
                    num_processes=num_processes,
                    file_timeout=file_timeout,
                    batch_size=batch_size
                )
                results.append(result)
                
                # Force garbage collection between species
                gc.collect()
                
            except Exception as e:
                print(f"Error processing species {species_id}: {e}")
                traceback.print_exc()
                results.append((species_id, False, f"Error: {str(e)}", 0))
    
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
    except Exception as e:
        print(f"Critical error in augmentation process: {str(e)}")
        traceback.print_exc()
        return []


def main():
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Robust audio augmentation for BirdCLEF dataset with batch-level parallelism")
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
        parser.add_argument("--processes", type=int, default=4,
                            help="Maximum number of parallel processes for file processing")
        parser.add_argument("--file_timeout", type=int, default=60,
                            help="Maximum time (seconds) to spend on a single file before timing out")
        parser.add_argument("--batch_size", type=int, default=5,
                            help="Number of files to process in each batch")
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
    
        # Process all species
        results = create_augmentations(
            species_list=species_to_augment,
            train_dir=args.train_dir,
            min_duration_seconds=args.min_duration,
            background_dir=args.background_dir,
            augmentations_per_file=args.augmentations_per_file,
            voice_data_path=args.voice_data_path,
            stats_csv=args.stats_csv,
            min_output_duration=args.min_output_duration,
            num_processes=args.processes,
            file_timeout=args.file_timeout,
            batch_size=args.batch_size
        )
    
        # Final summary
        print("\n=== FINAL AUGMENTATION SUMMARY ===")
        success_count = sum(1 for _, success, _, _ in results if success)
        print(f"Successfully processed {success_count} out of {len(species_to_augment)} species.")
    except Exception as e:
        print(f"Critical error in main function: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        traceback.print_exc()
