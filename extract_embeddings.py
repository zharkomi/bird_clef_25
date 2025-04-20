import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa
import soundfile as sf

from src.audio import parse_file
from src.birdnet import get_analyzer, analyze_chunk, get_embedding
from src.new_species import SPECIES


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


def split_into_chunks(audio, sr, chunk_duration=3.0, overlap=1.5):
    """
    Split audio into overlapping chunks of specified duration.

    Args:
        audio: Audio array
        sr: Sample rate
        chunk_duration: Duration of each chunk in seconds
        overlap: Overlap between chunks in seconds

    Returns:
        List of audio chunks
    """
    # Calculate parameters in samples
    chunk_size = int(chunk_duration * sr)
    hop_size = int((chunk_duration - overlap) * sr)

    # Calculate number of chunks
    num_chunks = max(1, 1 + int((len(audio) - chunk_size) / hop_size))

    chunks = []
    for i in range(num_chunks):
        start = i * hop_size
        end = start + chunk_size

        # If last chunk, handle potential boundary issues
        if end > len(audio):
            # Pad with zeros if needed
            chunk = np.zeros(chunk_size)
            chunk[:len(audio) - start] = audio[start:]
        else:
            chunk = audio[start:end]

        # Only add chunks that have sufficient non-zero content
        if np.count_nonzero(chunk) > 0.1 * len(chunk):
            chunks.append(chunk)

    return chunks


def process_species_audio(species_list=SPECIES,
                          train_dir="/home/mikhail/prj/bc_25_data/train_audio",
                          output_dir="/home/mikhail/prj/bird_clef_25/embeddings",
                          voice_data_path="/home/mikhail/prj/bird_clef_25/data/train_voice_data.pkl",
                          skip_existing=True):
    """
    Process audio files for all species in the list:
    1. Load voice data
    2. For each species, iterate over audio files
    3. Remove human voice segments
    4. Split into overlapping chunks
    5. Extract embeddings from each chunk
    6. Save embeddings

    Args:
        species_list: List of species IDs to process
        train_dir: Directory containing audio files organized by species
        output_dir: Directory to save embeddings
        voice_data_path: Path to pickle file with voice segments
        skip_existing: If True, skip files that already have embeddings saved
    """
    # Load voice data
    voice_data = load_voice_data(voice_data_path)

    # Initialize BirdNET analyzer
    get_analyzer()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each species
    for species_id in tqdm(species_list, desc="Processing species"):
        species_dir = os.path.join(train_dir, species_id)

        # Skip if species directory doesn't exist
        if not os.path.isdir(species_dir):
            print(f"Warning: Directory for species {species_id} not found at {species_dir}")
            continue

        # Create output directory for this species
        species_output_dir = os.path.join(output_dir, species_id)
        os.makedirs(species_output_dir, exist_ok=True)

        # Get all audio files for this species
        audio_files = [f for f in os.listdir(species_dir) if f.endswith('.ogg')]

        if not audio_files:
            print(f"Warning: No audio files found for species {species_id}")
            continue

        # Process each audio file
        for file_name in tqdm(audio_files, desc=f"Processing files for {species_id}"):
            file_path = os.path.join(species_dir, file_name)

            # Check if embeddings already exist for this file
            output_file = os.path.join(species_output_dir, f"{os.path.splitext(file_name)[0]}_embeddings.pkl")
            if skip_existing and os.path.exists(output_file):
                print(f"Skipping {file_name} - embeddings already exist")
                continue

            try:
                # Parse audio file
                sr, audio = parse_file(file_path)

                # Check if this file has voice segments
                full_path = os.path.join(species_id, file_name)

                voice_segments = []
                # Check different possible keys for the voice data
                key_format = f'/kaggle/input/birdclef-2025/train_audio/{full_path}'
                if key_format in voice_data:
                    voice_segments = voice_data[key_format]

                # Remove voice segments if present
                if voice_segments:
                    print(f"Removing {len(voice_segments)} voice segments from {file_name}")
                    audio = remove_voice_segments(audio, sr, voice_segments)

                # Split into overlapping chunks
                chunks = split_into_chunks(audio, sr)

                # Get embeddings for each chunk
                embeddings = []
                for i, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk, sr)
                    if embedding is not None and len(embedding) > 0:
                        embeddings.append({
                            'chunk_id': i,
                            'file_name': file_name,
                            'embedding': embedding
                        })

                # Save embeddings for this file
                if embeddings:
                    with open(output_file, 'wb') as f:
                        pickle.dump(embeddings, f)
                    print(f"Saved {len(embeddings)} embeddings for {file_name}")
                else:
                    print(f"Warning: No valid embeddings generated for {file_name}")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue


if __name__ == "__main__":
    # Set paths
    TRAIN_DIR = "/home/mikhail/prj/bc_25_data/train_audio"
    OUTPUT_DIR = "/home/mikhail/prj/bird_clef_25/embeddings"
    VOICE_DATA_PATH = "/home/mikhail/prj/bird_clef_25/data/train_voice_data.pkl"

    all_classes = os.listdir(TRAIN_DIR)
    # Process all species in the SPECIES list
    process_species_audio(
        species_list=all_classes,  # SPECIES,
        train_dir=TRAIN_DIR,
        output_dir=OUTPUT_DIR,
        voice_data_path=VOICE_DATA_PATH,
        skip_existing=True
    )
