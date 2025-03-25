import os

import pandas as pd

from audio import save_audio
from trim import trim_silence_adaptive
from wavelet import save_denoised, wavelet_denoise


def get_data_path():
    # Try to get the path from environment variable
    env_path = os.environ.get('DATA_PATH') + "/"
    # If environment variable exists and is not empty, use it
    if env_path and env_path.strip():
        return env_path
    # Otherwise, return the current directory path
    return os.getcwd()


def denoise_and_save(file_path, denoise_method, plot=False):
    sr, denoised = wavelet_denoise(
        file_path,
        denoise_method=denoise_method,
        n_noise_layers=5,  # Number of noise layers to remove
        wavelet='db8',  # Wavelet type
        level=1,  # Decomposition level
        plot=plot
    )
    # Save the denoised audio
    file_path += "." + denoise_method
    denoised_file_path = save_denoised(denoised, sr, file_path)
    print(f"Denoised audio saved to: {denoised_file_path}")
    return denoised


# Example usage
if __name__ == "__main__":
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(get_data_path() + 'train.csv')
    rating_counts = df['rating'].value_counts().sort_index()
    print("Count of items for each rating value:")
    print(rating_counts)

    rating_5_rows = df[df['rating'] == 5.0]
    print("First 10 rows with rating 5.0:")
    for filename in rating_5_rows['filename'].head(10):
        print(filename)

    # Load the audio file
    file_path = get_data_path() + "train_audio/21211/XC934741.ogg"

    # Process with wavelet transformation and plot
    # denoise_and_save(file_path, 'dwt')
    # denoise_and_save(file_path, 'emd')
    denoised = denoise_and_save(file_path, 'wpt')
    trimmed = trim_silence_adaptive(denoised)
    save_audio(trimmed, 32000, file_path + ".trimmed.mp3", "mp3")
