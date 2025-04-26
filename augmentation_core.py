import os
import random
import pickle
import numpy as np
import pandas as pd
import soundfile as sf
import time
import traceback
from multiprocessing import Process, Queue
from tqdm import tqdm

from audiomentations import (
    Compose, AddGaussianNoise, TimeStretch, PitchShift,
    Shift, Gain, RoomSimulator
)

from src.audio import parse_file
from file_statistics import calculate_total_duration


def load_voice_data(pkl_path):
    """
    Load the pickle file containing human voice segments.

    Args:
        pkl_path: Path to the pickle file

    Returns:
        Dictionary mapping filenames to voice segments
    """
    try:
        with open(pkl_path, 'rb') as f:
            voice_data = pickle.load(f)
        
        print(f"Loaded voice data for {len(voice_data)} files")
        return voice_data
    except Exception as e:
        print(f"Error loading voice data from {pkl_path}: {str(e)}")
        print(traceback.format_exc())
        return {}


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
    try:
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
    except Exception as e:
        print(f"Error removing voice segments: {str(e)}")
        print(traceback.format_exc())
        # Return the original audio if there's an error
        return audio


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
    try:
        voice_segments = []
        # Check different possible keys for the voice data
        key_format = f'/kaggle/input/birdclef-2025/train_audio/{full_path}'
        
        print(f"Looking for voice data with key: {key_format}")
        if key_format in voice_data:
            voice_segments = voice_data[key_format]
            print(f"Found {len(voice_segments)} voice segments for {file_name}")
        else:
            print(f"No voice segments found for {file_name} with key {key_format}")
            
        # Remove voice segments if present
        if voice_segments:
            print(f"Removing {len(voice_segments)} voice segments from {file_name}")
            audio = remove_voice_segments(audio, sr, voice_segments)
            print(f"After voice removal, audio duration: {len(audio) / sr:.2f} seconds")
        return audio
    except Exception as e:
        print(f"Error in voice removal for {file_name}: {str(e)}")
        print(traceback.format_exc())
        return audio


def add_custom_background_noise(audio, sr, background_dir, min_snr_db=3.0, max_snr_db=30.0):
    """
    Custom implementation of background noise addition that doesn't rely on AddBackgroundNoise class
    which has pickling issues with multiprocessing.
    
    Args:
        audio: Audio array
        sr: Sample rate
        background_dir: Directory with background noise files
        min_snr_db: Minimum signal-to-noise ratio in dB
        max_snr_db: Maximum signal-to-noise ratio in dB
        
    Returns:
        Audio with added background noise
    """
    try:
        if not background_dir or not os.path.exists(background_dir):
            print("No valid background directory provided, returning original audio")
            return audio
            
        # Find all background audio files
        bg_files = []
        for root, _, files in os.walk(background_dir):
            for file in files:
                if file.endswith(('.wav', '.ogg', '.mp3')):
                    bg_files.append(os.path.join(root, file))
        
        if not bg_files:
            print("No background files found, returning original audio")
            return audio
            
        # Choose a random background file
        bg_file = random.choice(bg_files)
        print(f"Using background file: {bg_file}")
        
        # Load the background audio
        try:
            bg_sr, bg_audio = parse_file(bg_file)
            print(f"Loaded background: length={len(bg_audio)}, sr={bg_sr}")
            
            # Resample if needed
            if bg_sr != sr:
                from librosa import resample
                bg_audio = resample(y=bg_audio, orig_sr=bg_sr, target_sr=sr)
                print(f"Resampled background to {sr}Hz")
                
            # Handle case where background is shorter than audio
            if len(bg_audio) < len(audio):
                # Repeat the background to match or exceed the length of the audio
                repetitions = int(np.ceil(len(audio) / len(bg_audio)))
                bg_audio = np.tile(bg_audio, repetitions)
                print(f"Extended background with {repetitions} repetitions")
                
            # Trim background to match audio length
            bg_audio = bg_audio[:len(audio)]
            
            # Calculate signal and noise power
            signal_power = np.mean(audio ** 2)
            noise_power = np.mean(bg_audio ** 2)
            
            # If either audio is silent, return the original
            if signal_power == 0 or noise_power == 0:
                print("Either signal or noise has zero power, returning original")
                return audio
                
            # Calculate current SNR in dB
            current_snr_db = 10 * np.log10(signal_power / noise_power)
            
            # Choose target SNR
            target_snr_db = random.uniform(min_snr_db, max_snr_db)
            print(f"Current SNR: {current_snr_db:.2f} dB, Target SNR: {target_snr_db:.2f} dB")
            
            # Calculate the gain to apply to the background noise
            snr_ratio = 10 ** ((current_snr_db - target_snr_db) / 20)
            bg_audio = bg_audio * snr_ratio
            
            # Add the background noise to the audio
            mixed_audio = audio + bg_audio
            
            # Normalize if needed to prevent clipping
            max_val = np.max(np.abs(mixed_audio))
            if max_val > 1.0:
                mixed_audio = mixed_audio / max_val
                print(f"Normalized audio to prevent clipping (max value was {max_val:.4f})")
                
            print("Successfully added background noise")
            return mixed_audio
            
        except Exception as e:
            print(f"Error processing background file: {str(e)}")
            print(traceback.format_exc())
            return audio
            
    except Exception as e:
        print(f"Error in custom background noise addition: {str(e)}")
        print(traceback.format_exc())
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
    try:
        # Calculate current duration in seconds
        current_duration = len(audio) / sr
    
        # If duration is already sufficient, return the original audio
        if current_duration >= min_duration:
            print(f"Audio already meets minimum duration: {current_duration:.2f}s >= {min_duration:.2f}s")
            return audio
    
        # Calculate how much audio to add in seconds
        additional_duration_needed = min_duration - current_duration
        print(f"Extending audio: current={current_duration:.2f}s, adding={additional_duration_needed:.2f}s")
    
        # If background directory is provided and exists, use background sounds
        if background_dir and os.path.exists(background_dir):
            print(f"Using background sounds from {background_dir} for extension")
            # Get all background files
            bg_files = []
            for root, _, files in os.walk(background_dir):
                for file in files:
                    if file.endswith(('.wav', '.ogg', '.mp3')):
                        bg_files.append(os.path.join(root, file))
    
            if bg_files:
                print(f"Found {len(bg_files)} background files")
                # Randomly select a background file
                bg_file = random.choice(bg_files)
                print(f"Selected background file: {bg_file}")
                try:
                    # Load background audio
                    print(f"Loading background file: {bg_file}")
                    bg_sr, bg_audio = parse_file(bg_file)
                    print(f"Background audio loaded: duration={len(bg_audio)/bg_sr:.2f}s, sr={bg_sr}Hz")
    
                    # Resample background audio if needed
                    if bg_sr != sr:
                        print(f"Resampling background from {bg_sr}Hz to {sr}Hz")
                        from librosa import resample
                        bg_audio = resample(y=bg_audio, orig_sr=bg_sr, target_sr=sr)
                        print(f"Resampled background duration: {len(bg_audio)/sr:.2f}s")
    
                    # Ensure background audio is long enough
                    if len(bg_audio) < int(additional_duration_needed * sr):
                        print(f"Background audio too short, repeating")
                        # Repeat background audio if necessary
                        repetitions = int(np.ceil((additional_duration_needed * sr) / len(bg_audio)))
                        bg_audio = np.tile(bg_audio, repetitions)
                        print(f"Extended background duration: {len(bg_audio)/sr:.2f}s")
    
                    # Trim background audio to required length
                    bg_audio = bg_audio[:int(additional_duration_needed * sr)]
                    print(f"Trimmed background duration: {len(bg_audio)/sr:.2f}s")
    
                    # Apply volume reduction to background audio
                    bg_volume_factor = 0.3  # 30% of original volume
                    bg_audio = bg_audio * bg_volume_factor
                    print(f"Applied volume reduction to background")
    
                    # Concatenate original audio with background
                    extended_audio = np.concatenate([audio, bg_audio])
                    print(f"Final extended audio duration: {len(extended_audio)/sr:.2f}s")
                    return extended_audio
    
                except Exception as e:
                    print(f"Error using background audio: {str(e)}")
                    print(traceback.format_exc())
                    # Fall back to repeating original audio if background fails
                    print("Falling back to repeating original audio")
    
        # If no background directory or an error occurred, repeat the original audio
        if len(audio) == 0:
            print("Original audio is empty, generating silence")
            # If audio is empty (e.g., after voice removal), generate silence
            extended_audio = np.zeros(int(min_duration * sr), dtype=np.float32)
        else:
            print(f"Repeating original audio to reach {min_duration:.2f}s")
            # Calculate how many times to repeat the audio
            repetitions = int(np.ceil(min_duration / current_duration))
            extended_audio = np.tile(audio, repetitions)
            print(f"Created repeated audio with {repetitions} repetitions")
    
            # Trim to required length
            extended_audio = extended_audio[:int(min_duration * sr)]
            print(f"Trimmed to final duration: {len(extended_audio)/sr:.2f}s")
    
        return extended_audio
    except Exception as e:
        print(f"Error extending audio: {str(e)}")
        print(traceback.format_exc())
        # If there's an error, return the original audio
        return audio


def create_augmentation_presets(background_dir=None):
    """
    Create multiple augmentation presets without AddBackgroundNoise.

    Args:
        background_dir: Optional directory containing background noise files (not used)

    Returns:
        List of Compose objects with different augmentation settings
    """
    try:
        print("Creating augmentation presets")
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
    
        print(f"Created {len(presets)} base augmentation presets")
        print("Skipping background noise augmenter due to multiprocessing compatibility issues")
        
        print(f"Final number of augmentation presets: {len(presets)}")
        return presets
    except Exception as e:
        print(f"Error creating augmentation presets: {str(e)}")
        print(traceback.format_exc())
        # Return a basic preset as fallback
        return [Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.5),
            Gain(min_gain_db=-3, max_gain_db=3, p=0.5)
        ])]


def safe_augment_file(file_path, species_id, voice_data, augmentation_presets,
                      min_output_duration, background_dir, aug_file_idx, timeout=60):
    """
    Run augmentation directly without using multiprocessing to avoid pickling issues.

    Args:
        file_path: Path to the audio file
        species_id: Species ID
        voice_data: Dictionary with voice segments
        augmentation_presets: List of augmentation presets
        min_output_duration: Minimum duration for output file
        background_dir: Directory with background sounds
        aug_file_idx: Augmentation index
        timeout: Maximum time to wait for process (seconds)

    Returns:
        Dictionary with augmentation results or error
    """
    print(f"Starting augmentation for {os.path.basename(file_path)} (aug #{aug_file_idx})")
    try:
        # Load the audio file
        print(f"Loading audio file: {file_path}")
        sr, audio = parse_file(file_path)
        print(f"Loaded audio with shape {audio.shape}, sample rate {sr}Hz, duration {len(audio)/sr:.2f}s")

        # Remove human voice before augmentation
        full_path = os.path.join(species_id, os.path.basename(file_path))
        print(f"Removing voice segments for {full_path}")
        audio = remove_voice(audio, os.path.basename(file_path), full_path, sr, voice_data)
        print(f"After voice removal: audio shape {audio.shape}, duration {len(audio)/sr:.2f}s")

        # Check if audio is too short after voice removal
        current_duration = len(audio) / sr
        if current_duration < 0.5:  # If less than 0.5 seconds, skip
            print(f"Audio too short after voice removal (only {current_duration:.2f}s), skipping")
            return {
                'status': 'skipped',
                'file_name': os.path.basename(file_path),
                'reason': f"Too short after voice removal (only {current_duration:.2f}s)"
            }

        # If audio is less than min_output_duration, extend it
        if current_duration < min_output_duration:
            print(f"Extending audio to meet minimum duration {min_output_duration}s")
            audio = extend_short_audio(audio, sr, min_output_duration, background_dir)
            print(f"After extension: audio shape {audio.shape}, duration {len(audio)/sr:.2f}s")

        # Get file name from path
        file_name = os.path.basename(file_path)
        base_file_name = os.path.splitext(file_name)[0]

        # Choose a random augmentation preset
        preset_idx = random.randint(0, len(augmentation_presets) - 1)
        augmenter = augmentation_presets[preset_idx]
        print(f"Selected augmentation preset #{preset_idx}")

        # Apply augmentation
        print(f"Applying augmentation to audio with shape {audio.shape}, sr={sr}")
        augmented_audio = augmenter(samples=audio, sample_rate=sr)
        print(f"Augmentation applied, new audio shape: {augmented_audio.shape}")

        # Optionally add background noise as a separate step
        if background_dir and random.random() < 0.7:  # 70% chance to add background
            print("Adding custom background noise")
            augmented_audio = add_custom_background_noise(
                augmented_audio, 
                sr, 
                background_dir, 
                min_snr_db=3.0, 
                max_snr_db=30.0
            )

        # Check if augmented audio is too short and extend if needed
        aug_duration = len(augmented_audio) / sr
        print(f"Augmented audio duration: {aug_duration:.2f}s")
        if aug_duration < min_output_duration:
            print(f"Extending augmented audio to meet minimum duration {min_output_duration}s")
            augmented_audio = extend_short_audio(augmented_audio, sr, min_output_duration, background_dir)
            aug_duration = len(augmented_audio) / sr
            print(f"Extended audio duration: {aug_duration:.2f}s")

        # Determine output file name
        aug_file_name = f"{base_file_name}_aug_{aug_file_idx}.ogg"
        aug_file_path = os.path.join(os.path.dirname(file_path), aug_file_name)
        print(f"Saving augmented audio to {aug_file_path}")

        # Save the augmented audio
        sf.write(aug_file_path, augmented_audio, sr)
        print(f"Successfully saved augmented audio")

        return {
            'status': 'success',
            'file_name': file_name,
            'aug_file_name': aug_file_name,
            'duration': aug_duration
        }

    except Exception as e:
        error_msg = f"Error in augmentation: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {
            'status': 'error',
            'file_name': os.path.basename(file_path),
            'error': error_msg
        }


def process_single_species(species_id, train_dir, min_duration_seconds, background_dir, augmentations_per_file,
                           voice_data_path, min_output_duration, voice_data=None, file_timeout=60):
    """
    Process a single species for augmentation with increased fault tolerance.
    Uses direct processing instead of isolated processes to avoid pickling issues.

    Args:
        species_id: ID of the species to process
        train_dir: Base training directory
        min_duration_seconds: Minimum duration required for the species
        background_dir: Directory with background noises
        augmentations_per_file: Number of augmentations per original file
        voice_data_path: Path to voice segments data file
        min_output_duration: Minimum duration for each output file
        voice_data: If provided, use this voice data instead of loading from file (for single-process mode)
        file_timeout: Maximum time (seconds) to spend on a single file before timing out

    Returns:
        Tuple of (species_id, success, message, new_duration)
    """
    # Initialize process-specific random seed
    random.seed(os.getpid() + int(time.time() * 1000) % 10000)
    np.random.seed(os.getpid() + int(time.time() * 1000) % 10000)

    # Use process-local logging to avoid conflicts between processes
    def log_message(message):
        # Adding process ID ensures we can trace which process generated which message
        print(f"[Process {os.getpid()}, Species {species_id}] {message}")

    log_message(f"Starting processing for species {species_id}")

    # Load voice data in this process if not provided
    if voice_data is None:
        try:
            log_message(f"Loading voice data from {voice_data_path}")
            with open(voice_data_path, 'rb') as f:
                voice_data = pickle.load(f)
            log_message(f"Loaded voice data with {len(voice_data)} entries")
        except Exception as e:
            log_message(f"Error loading voice data: {e}")
            log_message(traceback.format_exc())
            voice_data = {}

    species_dir = os.path.join(train_dir, species_id)
    log_message(f"Species directory: {species_dir}")

    # Skip if species directory doesn't exist
    if not os.path.isdir(species_dir):
        log_message(f"Warning: Directory for species {species_id} not found at {species_dir}")
        return species_id, False, "Directory not found", 0

    # Calculate total duration
    log_message(f"Calculating total duration for {species_id}")
    total_duration = calculate_total_duration(species_dir)
    log_message(f"Species {species_id}: Total duration = {total_duration:.2f} seconds")

    # If duration is sufficient, skip to next species
    if total_duration >= min_duration_seconds:
        log_message(f"Species {species_id} has sufficient audio ({total_duration:.2f} seconds). Skipping augmentation.")
        return species_id, True, "Already sufficient", total_duration

    # Get all audio files for this species, excluding already augmented files
    log_message(f"Finding original audio files for {species_id}")
    audio_files = [f for f in os.listdir(species_dir) if f.endswith('.ogg') and "_aug_" not in f]
    log_message(f"Found {len(audio_files)} original audio files")

    if not audio_files:
        log_message(f"Warning: No original audio files found for species {species_id}")
        return species_id, False, "No original audio files found", 0

    # Calculate how much more audio is needed
    additional_duration_needed = min_duration_seconds - total_duration
    log_message(f"Need to generate {additional_duration_needed:.2f} more seconds for {species_id}")

    # Track progress
    additional_duration_generated = 0
    augmentation_count = 0
    success_count = 0
    failure_count = 0
    skip_count = 0

    # Create augmentation presets
    log_message(f"Creating augmentation presets")
    augmentation_presets = create_augmentation_presets(None)  # We're not using AddBackgroundNoise
    log_message(f"Created {len(augmentation_presets)} augmentation presets")

    # Create a queue of files to process
    files_to_process = []
    avg_file_duration = total_duration / len(audio_files) if len(audio_files) > 0 else 1.0
    target_files = int(np.ceil(additional_duration_needed / (avg_file_duration * augmentations_per_file)))
    log_message(f"Targeting approximately {target_files} files to process (avg duration: {avg_file_duration:.2f}s)")
    
    # Fill the queue with potentially repeated files
    while len(files_to_process) < target_files:
        files_to_process.extend(audio_files)
    
    # Trim to target size and shuffle
    files_to_process = files_to_process[:target_files]
    random.shuffle(files_to_process)
    log_message(f"Created processing queue with {len(files_to_process)} files")

    # Track existing augmentation files
    existing_aug_files = set([f for f in os.listdir(species_dir) if f.endswith('.ogg') and "_aug_" in f])
    log_message(f"Found {len(existing_aug_files)} existing augmentation files")

    # Count duration of existing augmentation files
    log_message(f"Calculating duration of existing augmentation files")
    for aug_file in existing_aug_files:
        try:
            aug_file_path = os.path.join(species_dir, aug_file)
            sr, audio = parse_file(aug_file_path)
            existing_duration = len(audio) / sr
            additional_duration_generated += existing_duration
            augmentation_count += 1
        except Exception as e:
            log_message(f"Error reading existing augmentation file {aug_file}: {e}")
            log_message(traceback.format_exc())

    log_message(f"Existing augmentation files contribute {additional_duration_generated:.2f} seconds")

    # Process each file in the queue
    for file_idx, file_name in enumerate(files_to_process):
        if additional_duration_generated >= additional_duration_needed:
            log_message(f"Target duration reached, stopping augmentation")
            break

        file_path = os.path.join(species_dir, file_name)
        base_file_name = os.path.splitext(file_name)[0]
        log_message(f"Processing file {file_idx+1}/{len(files_to_process)}: {file_name}")

        # Try augmenting this file multiple times
        for aug_idx in range(augmentations_per_file):
            if additional_duration_generated >= additional_duration_needed:
                log_message(f"Target duration reached during augmentation, stopping")
                break

            # Define augmentation file name
            aug_file_idx = augmentation_count + 1
            aug_file_name = f"{base_file_name}_aug_{aug_file_idx}.ogg"
            aug_file_path = os.path.join(species_dir, aug_file_name)

            # Skip if this augmentation already exists
            if aug_file_name in existing_aug_files or os.path.exists(aug_file_path):
                log_message(f"Augmentation file {aug_file_name} already exists, skipping creation")
                continue

            # Run augmentation directly in current process
            log_message(f"Processing {file_name} (aug #{aug_file_idx})")
            result = safe_augment_file(
                file_path=file_path,
                species_id=species_id,
                voice_data=voice_data,
                augmentation_presets=augmentation_presets,
                min_output_duration=min_output_duration,
                background_dir=background_dir,
                aug_file_idx=aug_file_idx,
                timeout=file_timeout
            )

            # Process result
            if result['status'] == 'success':
                success_count += 1
                augmentation_count += 1
                additional_duration_generated += result['duration']
                log_message(f"Generated {result['aug_file_name']} ({result['duration']:.2f} seconds)")
            elif result['status'] == 'skipped':
                skip_count += 1
                log_message(f"Skipped {file_name}: {result['reason']}")
            else:
                failure_count += 1
                log_message(f"Error processing {file_name}: {result['error']}")

    # Calculate new total duration
    log_message(f"Calculating final duration after augmentation")
    new_total_duration = calculate_total_duration(species_dir)
    log_message(f"Species {species_id}: New total duration = {new_total_duration:.2f} seconds")
    log_message(f"Summary: {success_count} successful, {failure_count} failed, {skip_count} skipped")

    return species_id, True, f"Generated {augmentation_count} augmentations", new_total_duration