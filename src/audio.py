import os
import tempfile

import librosa
import librosa.display
import numpy as np
import pandas as pd
import soundfile as sf
from pydub import AudioSegment


def parse_file(path_to_file):
    # Load and parse OGG file
    print(f"Parsing {path_to_file} ")
    # y, sr = librosa.load(path_to_file, sr=32000)
    y = librosa.load(
        path_to_file, sr=48000, mono=True, res_type="kaiser_fast"
    )
    return 48000, y[0]


def save_audio(signal, sr, file_path, format):
    """
    Save audio signal to file

    Parameters:
        signal (numpy.ndarray): Audio signal to save
        sr (int): Sample rate
        file_path (str): Path to save the audio file
        format (str): Audio format
    """
    print("saving audio to", file_path)
    if format.lower() == 'mp3':
        # For MP3 format, use pydub
        try:
            signal = signal / np.max(np.abs(signal))
            # Create a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                temp_wav_path = temp_wav.name
            # Write the signal to the temporary WAV file
            sf.write(temp_wav_path, signal, sr, format='WAV')
            # Convert to MP3 using pydub from the temporary file
            audio_segment = AudioSegment.from_wav(temp_wav_path)
            # Clean up the temporary file after conversion
            os.unlink(temp_wav_path)
            audio_segment.export(file_path, format="mp3", bitrate="192k")
        except ImportError:
            print("MP3 export requires pydub and ffmpeg. Install with:")
            print("pip install pydub")
            print("Plus ffmpeg must be installed on your system")
            # Fallback to WAV if pydub or ffmpeg is not available
            file_path = file_path.replace('.mp3', '.wav')
            sf.write(file_path, signal, sr)
    else:
        # For other formats, use soundfile
        sf.write(file_path, signal, sr, format=format.upper())
    # print(f"Saved denoised audio to {file_path}")
