import os
import numpy as np
import librosa
import pickle
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from birdnetlib import RecordingBuffer
from tqdm import tqdm
import glob

# Path to your audio files
AUDIO_DIR = '/home/mikhail/prj/bc_25_data/train_audio'

# Output directory for embeddings
OUTPUT_DIR = '/home/mikhail/prj/bird_clef_25/embeddings'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Function to get BirdNET analyzer instance
def get_analyzer():
    # Initialize the analyzer with default settings
    # The BirdNET model expects 48kHz audio
    return Analyzer(sampling_rate=48000)


# Function to analyze audio chunk and get embeddings
def analyze_chunk(chunk, sample_rate):
    # Create a RecordingBuffer object for this chunk
    # RecordingBuffer is designed to work with raw audio buffers
    analyzer = get_analyzer()
    recording = RecordingBuffer(
        analyzer,
        buffer=chunk,
        rate=sample_rate,
        return_all_detections=True
    )
    # Process the audio data
    recording.analyze()
    return recording


# Function to extract embeddings from an audio file
def extract_embeddings(audio_path):
    try:
        # Load audio file at original sample rate
        audio, sr = librosa.load(audio_path, sr=None)

        # Resample to 48000 Hz (BirdNET requirement)
        if sr != 48000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
            sr = 48000

        # Calculate chunk size in samples (5 seconds * sample rate)
        chunk_size = 5 * sr

        # Get total number of complete chunks
        num_chunks = len(audio) // chunk_size

        # Storage for embeddings from all chunks
        all_embeddings = []

        # Process each 5-second chunk
        for i in range(num_chunks):
            # Extract the chunk
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            chunk = audio[start_idx:end_idx]

            # Analyze the chunk
            recording = analyze_chunk(chunk, sr)

            # Get embeddings from the recording
            # Note: This assumes that birdnetlib RecordingBuffer provides access to embeddings
            chunk_embeddings = recording.embeddings

            # Store embeddings if valid
            if chunk_embeddings is not None and len(chunk_embeddings) > 0:
                all_embeddings.append(chunk_embeddings)

        # If no embeddings were extracted from any chunk, return None
        if not all_embeddings:
            print(f"Warning: No embeddings extracted for {audio_path}")
            return None

        # Flatten list of embeddings from all chunks
        flat_embeddings = np.vstack(all_embeddings)

        return flat_embeddings

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def process_species_directory(species_id):
    """Process all audio files for a given species"""
    species_dir = os.path.join(AUDIO_DIR, species_id)
    output_species_dir = os.path.join(OUTPUT_DIR, species_id)

    # Skip if not a directory
    if not os.path.isdir(species_dir):
        return

    # Create output directory for this species
    os.makedirs(output_species_dir, exist_ok=True)

    # Get all OGG files for this species
    audio_files = glob.glob(os.path.join(species_dir, "*.ogg"))

    if not audio_files:
        print(f"No audio files found for species {species_id}")
        return

    print(f"Processing {len(audio_files)} files for species {species_id}")

    # Process each audio file
    for audio_file in tqdm(audio_files, desc=f"Species {species_id}"):
        # Generate output filename based on the audio filename
        base_name = os.path.basename(audio_file).replace('.ogg', '')
        output_file = os.path.join(output_species_dir, f"{base_name}_embeddings.pkl")

        # Skip if already processed
        if os.path.exists(output_file):
            continue

        # Extract embeddings
        embeddings = extract_embeddings(audio_file)

        # Save embeddings if successful
        if embeddings is not None and len(embeddings) > 0:
            with open(output_file, 'wb') as f:
                pickle.dump(embeddings, f)


def main():
    # Get list of all species directories
    species_dirs = [d for d in os.listdir(AUDIO_DIR) if os.path.isdir(os.path.join(AUDIO_DIR, d))]

    print(f"Found {len(species_dirs)} species directories")

    # Process each species directory
    for species_id in species_dirs:
        process_species_directory(species_id)

    print("Embedding extraction complete!")


if __name__ == "__main__":
    main()