import matplotlib.pyplot as plt

from audio import parse_file, save_audio
from PyEMD import EMD as PyEMD  # Import the EMD implementation


def remove_noise_layers(coeffs, n_layers_to_remove=1, threshold_method='soft', threshold_factor=1.0):
    """
    Remove noise from wavelet coefficients by thresholding detail coefficients

    Parameters:
        coeffs (list): Wavelet coefficients from wavedec
        n_layers_to_remove (int): Number of detail coefficient layers to denoise
        threshold_method (str): 'soft' or 'hard' thresholding
        threshold_factor (float): Multiplier for threshold calculation

    Returns:
        tuple: (denoised_coeffs, noise_coeffs) - Denoised and removed noise coefficients
    """
    # Make a copy of coefficients for denoising
    denoised_coeffs = list(coeffs)
    noise_coeffs = list(coeffs)

    # Initialize noise coefficients with zeros for approximation coefficients
    noise_coeffs[0] = np.zeros_like(coeffs[0])

    # Process detail coefficients (starting from highest frequency)
    for i in range(1, min(n_layers_to_remove + 1, len(coeffs))):
        # Calculate threshold using median absolute deviation
        detail = coeffs[-i]
        threshold = threshold_factor * np.sqrt(2 * np.log(len(detail))) * np.median(np.abs(detail)) / 0.6745

        # Store original coefficients in noise_coeffs
        noise_coeffs[-i] = coeffs[-i].copy()

        if threshold_method == 'hard':
            # Hard thresholding: set values below threshold to zero
            denoised_coeffs[-i] = pywt.threshold(detail, threshold, mode='hard')
            # Calculate noise (what was removed)
            noise_coeffs[-i] = detail - denoised_coeffs[-i]
        else:
            # Soft thresholding: shrink values above threshold
            denoised_coeffs[-i] = pywt.threshold(detail, threshold, mode='soft')
            # Calculate noise (what was removed)
            noise_coeffs[-i] = detail - denoised_coeffs[-i]

    # Set the remaining detail coefficients to zero in noise_coeffs
    for i in range(n_layers_to_remove + 1, len(coeffs)):
        noise_coeffs[-i] = np.zeros_like(coeffs[-i])

    return denoised_coeffs, noise_coeffs


def remove_noise_layers_wpt(wpt, n_layers_to_remove=1, threshold_method='soft', threshold_factor=1.0):
    """
    Remove noise from wavelet packet tree by thresholding the detail coefficients

    Parameters:
        wpt (WaveletPacket): Wavelet Packet object
        n_layers_to_remove (int): Number of detail coefficient layers to denoise
        threshold_method (str): 'soft' or 'hard' thresholding
        threshold_factor (float): Multiplier for threshold calculation

    Returns:
        wpt_denoised: Denoised Wavelet Packet object
        noise_signal: Noise signal (difference between original and denoised)
    """
    # Initialize a new WaveletPacket object for denoised signal
    wpt_denoised = pywt.WaveletPacket(data=wpt.data, wavelet=wpt.wavelet, mode=wpt.mode)

    # Loop through the detail nodes at each level, from highest frequency
    for level in range(1, n_layers_to_remove + 1):
        # Get all nodes at this level (both approximation and detail)
        nodes_at_level = wpt.get_level(level)

        for node in nodes_at_level:
            # Only process detail nodes (not approximation nodes)
            if node.path[-1] != 'a':  # 'a' indicates approximation, we skip those
                detail = node.data
                # Calculate threshold using median absolute deviation
                threshold = threshold_factor * np.sqrt(2 * np.log(len(detail))) * np.median(np.abs(detail)) / 0.6745

                # Thresholding coefficients
                if threshold_method == 'hard':
                    denoised_data = pywt.threshold(detail, threshold, mode='hard')
                else:
                    denoised_data = pywt.threshold(detail, threshold, mode='soft')

                # Set the denoised data in the wavelet packet tree
                wpt_denoised[node.path] = denoised_data

    # Reconstruct the noise signal by subtracting denoised data from original
    noise_signal = wpt.reconstruct(update=True) - wpt_denoised.reconstruct(update=True)

    return wpt_denoised, noise_signal


def denoise_imf_with_wavelets(imf, wavelet='db8', level=3, threshold_method='soft', threshold_factor=1.0):
    """
    Denoise Intrinsic Mode Function (IMF) using wavelet thresholding

    Parameters:
        imf (numpy.ndarray): IMF to denoise
        wavelet (str): Wavelet type to use
        level (int): Decomposition level
        threshold_method (str): 'soft' or 'hard' thresholding
        threshold_factor (float): Multiplier for threshold calculation

    Returns:
        numpy.ndarray: Denoised IMF
    """
    # Check if IMF has enough data points for the requested level
    if len(imf) < 2 ** (level + 1):
        # If IMF is too short, use a lower level
        adjusted_level = max(1, int(np.log2(len(imf))) - 1)
        print(f"Warning: IMF length {len(imf)} is too short for level {level}. Using level {adjusted_level} instead.")
        level = adjusted_level

    # Apply Discrete Wavelet Transform
    coeffs = pywt.wavedec(imf, wavelet, level=level)

    # Calculate signal energy for adaptive thresholding
    signal_energy = np.sum(imf ** 2)
    if signal_energy < 1e-10:  # Very low energy, likely no significant content
        return imf  # Return original without denoising

    # Remove noise from wavelet coefficients - first calculate target noise level
    denoised_coeffs, _ = remove_noise_layers(coeffs, n_layers_to_remove=level,
                                             threshold_method=threshold_method,
                                             threshold_factor=threshold_factor)

    # Reconstruct denoised IMF
    denoised_imf = pywt.waverec(denoised_coeffs, wavelet)

    # Ensure same length as original
    denoised_imf = denoised_imf[:len(imf)]

    # Check if denoising removed too much signal energy
    denoised_energy = np.sum(denoised_imf ** 2)
    if denoised_energy < 0.01 * signal_energy:  # Less than 1% energy remained
        # Blend original and denoised to preserve some signal characteristics
        blend_ratio = 0.7  # Keep 70% of original signal
        denoised_imf = blend_ratio * imf + (1 - blend_ratio) * denoised_imf

    return denoised_imf


def perform_emd_wavelet_denoising(y, n_imfs_to_denoise=3, wavelet='db8', level=3,
                                  threshold_method='soft', threshold_factor=1.0,
                                  denoise_strength=0.5, preserve_ratio=0.7):
    """
    Perform EMD followed by Wavelet denoising on selected IMFs

    Parameters:
        y (numpy.ndarray): Input signal
        n_imfs_to_denoise (int): Number of high-frequency IMFs to denoise
        wavelet (str): Wavelet type to use
        level (int): Decomposition level for wavelets
        threshold_method (str): 'soft' or 'hard' thresholding
        threshold_factor (float): Multiplier for threshold calculation
        denoise_strength (float): Strength of denoising from 0.0 (none) to 1.0 (full)
        preserve_ratio (float): How much of the original signal to blend back in (0.0-1.0)

    Returns:
        tuple: (denoised_signal, noise_signal, imfs) - Denoised signal, noise signal, and all IMFs
    """
    # Initialize EMD with optimized parameters
    emd = PyEMD()

    # For large signals, downsample to improve performance
    original_length = len(y)
    max_length = 10000  # Reasonable limit for EMD computation

    if original_length > max_length:
        # Downsample using simple decimation for performance
        decimation_factor = original_length // max_length + 1
        y_downsampled = y[::decimation_factor]
        print(f"Signal downsampled by factor {decimation_factor} for EMD calculation")
    else:
        y_downsampled = y
        decimation_factor = 1

    # Set limits to prevent infinite computation
    emd.MAX_ITERATION = 50  # Limit iterations per IMF

    try:
        # Extract IMFs with timeout protection
        print("Starting EMD decomposition...")
        imfs = emd.emd(y_downsampled, max_imf=5)  # Limiting to 5 IMFs for speed
        print(f"EMD completed: extracted {imfs.shape[0]} IMFs")

        # If we downsampled, we need to upsample IMFs back to original size
        if decimation_factor > 1:
            # Create empty array for upsampled IMFs
            upsampled_imfs = np.zeros((imfs.shape[0], original_length))

            # Upsample each IMF individually using linear interpolation
            for i in range(imfs.shape[0]):
                # Create indices for original and upsampled signals
                orig_indices = np.arange(0, original_length, decimation_factor)[:len(imfs[i])]
                target_indices = np.arange(original_length)

                # Use linear interpolation to upsample
                upsampled_imf = np.interp(target_indices, orig_indices, imfs[i])
                upsampled_imfs[i] = upsampled_imf

            imfs = upsampled_imfs
            print("IMFs upsampled to original signal length")

    except Exception as e:
        print(f"EMD computation error: {str(e)}")
        # Fallback: use wavelet decomposition instead
        print("Falling back to wavelet decomposition as alternative to EMD")

        # Create pseudo-IMFs using wavelet decomposition
        coeffs = pywt.wavedec(y, wavelet, level=5)
        imfs = np.zeros((len(coeffs), len(y)))

        # Convert each wavelet level to a pseudo-IMF
        for i, coeff in enumerate(coeffs):
            # For approximation coefficients, reconstruct directly
            if i == 0:
                rec_coeffs = [coeff] + [None] * (len(coeffs) - 1)
            # For detail coefficients, reconstruct one level at a time
            else:
                rec_coeffs = [None] * len(coeffs)
                rec_coeffs[i] = coeff

            # Reconstruct this component and make it the same length as original signal
            component = pywt.waverec(rec_coeffs, wavelet)
            imfs[i, :len(component)] = component[:len(y)]

        print(f"Created {len(coeffs)} pseudo-IMFs using wavelet decomposition")

    # Make a copy of IMFs for denoising
    denoised_imfs = imfs.copy()

    # Keep track of removed noise for each IMF
    noise_imfs = np.zeros_like(imfs)

    # Analyze IMF energy to determine adaptive thresholding
    imf_energies = np.array([np.sum(imf ** 2) for imf in imfs])
    total_energy = np.sum(imf_energies)

    # Normalize energies and determine importance
    if total_energy > 0:
        normalized_energies = imf_energies / total_energy
    else:
        normalized_energies = np.ones_like(imf_energies) / len(imf_energies)

    # Apply adaptive threshold factor based on energy distribution
    print("IMF energy distribution:", normalized_energies)

    # Denoise selected IMFs (usually the highest frequency ones)
    for i in range(min(n_imfs_to_denoise, imfs.shape[0])):
        # Apply wavelet denoising to each IMF
        original_imf = imfs[i].copy()

        # Adapt threshold based on IMF importance and user-specified denoise strength
        adaptive_threshold_factor = threshold_factor * denoise_strength

        # Higher frequency IMFs typically contain more noise, so we can be more aggressive with them
        # Lower the index, higher the frequency, more aggressive denoising
        imf_specific_factor = adaptive_threshold_factor * (1.0 - i / n_imfs_to_denoise * 0.5)

        # For first IMF (highest frequency), use more aggressive denoising
        if i == 0:
            # Higher threshold factor for first IMF which typically contains most noise
            denoised_imf = denoise_imf_with_wavelets(
                imfs[i], wavelet, level, threshold_method, imf_specific_factor * 1.5
            )
            # Apply more aggressive denoising strength for first IMF
            denoised_imf = (denoised_imf * min(0.9, denoise_strength * 1.3) +
                            original_imf * (1 - min(0.9, denoise_strength * 1.3)))
        else:
            # For other IMFs, use standard settings
            denoised_imf = denoise_imf_with_wavelets(
                imfs[i], wavelet, level, threshold_method, imf_specific_factor
            )

            # Apply partial denoising with diminishing strength for lower IMFs
            imf_denoise_strength = denoise_strength * (1.0 - 0.2 * i)
            denoised_imf = (denoised_imf * imf_denoise_strength +
                            original_imf * (1 - imf_denoise_strength))

        # Store denoised IMF
        denoised_imfs[i] = denoised_imf

        # Store noise that was removed
        noise_imfs[i] = original_imf - denoised_imf

    # Reconstruct signals
    denoised_signal = np.sum(denoised_imfs, axis=0)
    noise_signal = np.sum(noise_imfs, axis=0)

    # Blend in some of the original signal to preserve important audio features
    if preserve_ratio > 0:
        print(f"Preserving {preserve_ratio * 100:.1f}% of original signal characteristics")
        denoised_signal = denoised_signal * (1 - preserve_ratio) + y * preserve_ratio
        # Recalculate noise
        noise_signal = y - denoised_signal

    return denoised_signal, noise_signal, imfs


def plot_imfs(imfs, denoised_imfs, sr, n_imfs_to_denoise):
    """
    Plot original and denoised IMFs

    Parameters:
        imfs (numpy.ndarray): Original IMFs
        denoised_imfs (numpy.ndarray): Denoised IMFs
        sr (int): Sample rate
        n_imfs_to_denoise (int): Number of IMFs that were denoised
    """
    n_imfs = imfs.shape[0]
    time = np.arange(imfs.shape[1]) / sr

    plt.figure(figsize=(15, 3 * n_imfs))

    for i in range(n_imfs):
        plt.subplot(n_imfs, 2, 2 * i + 1)
        plt.plot(time, imfs[i])
        plt.title(f"Original IMF {i + 1}")
        plt.grid(True)

        plt.subplot(n_imfs, 2, 2 * i + 2)
        if i < n_imfs_to_denoise:
            plt.plot(time, denoised_imfs[i])
            plt.title(f"Denoised IMF {i + 1}")
        else:
            plt.plot(time, imfs[i])
            plt.title(f"IMF {i + 1} (Not Denoised)")
        plt.grid(True)

    plt.tight_layout()
    plt.show()


def save_denoised(signal, sr, original_file_path, format='mp3'):
    """
    Save denoised audio signal to a file with '_denoized' suffix

    Parameters:
        signal (numpy.ndarray): Denoised audio signal to save
        sr (int): Sample rate
        original_file_path (str): Original file path to derive the new filename
        format (str): Audio format to save as ('mp3', 'ogg', 'wav', etc.)

    Returns:
        str: Path of the saved file
    """
    denoised_file_path = original_file_path + f"_denoized.{format}"
    save_audio(signal, sr, denoised_file_path, format)
    return denoised_file_path


def plot_signals(y, denoised_signal, noise_signal, sr, file_path, n_noise_layers):
    """
    Plot original, denoised, and noise signals

    Parameters:
        y (numpy.ndarray): Original signal
        denoised_signal (numpy.ndarray): Denoised signal
        noise_signal (numpy.ndarray): Noise signal
        sr (int): Sample rate
        file_path (str): Path to the original audio file
        n_noise_layers (int): Number of detail coefficient layers denoised
    """
    # Time axis for plotting
    time = np.arange(len(y)) / sr

    # Plot original, denoised, and noise signals
    plt.figure(figsize=(15, 10))

    # Original signal
    plt.subplot(3, 1, 1)
    plt.plot(time, y)
    plt.title(f"Original Waveform of {os.path.basename(file_path)}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Denoised signal
    plt.subplot(3, 1, 2)
    plt.plot(time, denoised_signal)
    plt.title(f"Denoised Waveform (Removed {n_noise_layers} noise layers)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Extracted noise
    plt.subplot(3, 1, 3)
    plt.plot(time, noise_signal)
    plt.title("Extracted Noise")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_wavelet_coeffs(coeffs, denoised_coeffs, noise_coeffs, wavelet, level):
    """
    Plot wavelet coefficients before and after denoising

    Parameters:
        coeffs (list): Original wavelet coefficients
        denoised_coeffs (list): Denoised wavelet coefficients
        noise_coeffs (list): Noise wavelet coefficients
        wavelet (str): Wavelet type
        level (int): Decomposition level
    """
    plt.figure(figsize=(15, 10))

    # Plot approximation coefficients
    plt.subplot(3, 1, 1)
    plt.plot(coeffs[0])
    plt.title(f"Approximation Coefficients (Level {level})")
    plt.grid(True)

    # Plot detail coefficients - original vs denoised
    plt.subplot(3, 1, 2)

    # Concatenate all detail coefficients for visualization
    all_details = np.concatenate([coeff for coeff in coeffs[1:]])
    all_denoised_details = np.concatenate([coeff for coeff in denoised_coeffs[1:]])

    plt.plot(all_details, alpha=0.7, label='Original')
    plt.plot(all_denoised_details, alpha=0.7, label='Denoised')
    plt.title("Detail Coefficients: Original vs Denoised")
    plt.legend()
    plt.grid(True)

    # Plot noise coefficients
    plt.subplot(3, 1, 3)
    all_noise_details = np.concatenate([coeff for coeff in noise_coeffs[1:]])
    plt.plot(all_noise_details)
    plt.title("Extracted Noise Coefficients")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


import os
import pandas as pd
import pywt
import numpy as np


def generate_config_id(path_to_file, denoise_method, n_noise_layers, wavelet, level,
                       threshold_method, threshold_factor, denoise_strength, preserve_ratio):
    """
    Generate a unique identifier for a specific processing configuration.

    Parameters:
        path_to_file (str): Path to audio file
        denoise_method (str): Denoising method used
        n_noise_layers (int): Number of detail coefficient layers to denoise
        wavelet (str): Wavelet type used
        level (int): Decomposition level
        threshold_method (str): Thresholding method used
        threshold_factor (float): Threshold multiplier
        denoise_strength (float): Strength of denoising
        preserve_ratio (float): Ratio of original signal to preserve

    Returns:
        str: Unique configuration ID
    """
    return f"{os.path.basename(path_to_file)}_{denoise_method}_{n_noise_layers}_{wavelet}_{level}_{threshold_method}_{threshold_factor}_{denoise_strength}_{preserve_ratio}"


def get_csv_path(path_to_file, csv_path, denoise_method):
    """
    Determine the CSV file path based on the audio file path.

    Parameters:
        path_to_file (str): Path to audio file
        csv_path (str): Optional explicit CSV path

    Returns:
        str: Path to CSV file
    """
    if csv_path is None:
        csv_dir = os.path.dirname(path_to_file)
        file_name = os.path.splitext(os.path.basename(path_to_file))[0]
        return os.path.join(csv_dir, f"{file_name}_{denoise_method}.csv")
    return csv_path


def load_existing_results(csv_path, config_id):
    """
    Try to load existing results for a specific configuration.

    Parameters:
        csv_path (str): Path to CSV file
        config_id (str): Configuration ID to look for

    Returns:
        tuple: (found, sr, y, denoised_signal, noise_signal) - Whether results were found and the signals
    """
    if not os.path.exists(csv_path):
        print(f"No existing results file found at: {csv_path}")
        return False, None, None, None, None

    try:
        # Load the CSV file
        results_df = pd.read_csv(csv_path)

        # Check if this exact configuration exists
        if 'config_id' not in results_df.columns or config_id not in results_df['config_id'].values:
            print(f"No existing results for configuration: {config_id}")
            return False, None, None, None, None

        # Get the row for this configuration
        result_row = results_df[results_df['config_id'] == config_id].iloc[0]

        # Check if the necessary columns exist
        if not all(col in results_df.columns for col in ['sr', 'original', 'denoised', 'noise']):
            print("Existing results file format is incompatible.")
            return False, None, None, None, None

        print("Loading existing results instead of reprocessing...")
        sr = result_row['sr']

        # Extract the signals from the CSV
        y = np.fromstring(result_row['original'].strip('[]'), sep=',')
        denoised_signal = np.fromstring(result_row['denoised'].strip('[]'), sep=',')
        noise_signal = np.fromstring(result_row['noise'].strip('[]'), sep=',')

        print(f"Successfully loaded existing results with {len(y)} samples")
        return True, sr, y, denoised_signal, noise_signal

    except Exception as e:
        print(f"Error loading existing results: {e}")
        return False, None, None, None, None


def process_with_dwt(y, wavelet, level, n_noise_layers, threshold_method, threshold_factor, plot=False):
    """
    Process signal using Discrete Wavelet Transform.

    Parameters:
        y (array): Input signal
        wavelet (str): Wavelet type
        level (int): Decomposition level
        n_noise_layers (int): Number of detail layers to denoise
        threshold_method (str): Thresholding method
        threshold_factor (float): Threshold multiplier
        plot (bool): Whether to plot coefficients

    Returns:
        tuple: (denoised_signal, noise_signal)
    """
    # Apply Discrete Wavelet Transform
    coeffs = pywt.wavedec(y, wavelet, level=level)

    # Remove noise layers
    denoised_coeffs, noise_coeffs = remove_noise_layers(
        coeffs, n_noise_layers, threshold_method, threshold_factor
    )

    # Reconstruct signals
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
    noise_signal = pywt.waverec(noise_coeffs, wavelet)

    # Plot wavelet coefficients if requested
    if plot:
        plot_wavelet_coeffs(coeffs, denoised_coeffs, noise_coeffs, wavelet, level)

    return denoised_signal, noise_signal


def process_with_wpt(y, wavelet, n_noise_layers, threshold_method, threshold_factor):
    """
    Process signal using Wavelet Packet Transform.

    Parameters:
        y (array): Input signal
        wavelet (str): Wavelet type
        n_noise_layers (int): Number of detail layers to denoise
        threshold_method (str): Thresholding method
        threshold_factor (float): Threshold multiplier

    Returns:
        tuple: (denoised_signal, noise_signal)
    """
    # Apply Wavelet Packet Transform
    wpt = pywt.WaveletPacket(data=y, wavelet=wavelet, mode='symmetric')

    # Remove noise from WPT
    wpt_denoised, noise_signal = remove_noise_layers_wpt(
        wpt, n_noise_layers, threshold_method, threshold_factor
    )

    # Reconstruct denoised signal from WPT
    denoised_signal = wpt_denoised.reconstruct(update=True)

    return denoised_signal, noise_signal


def process_with_emd(y, n_noise_layers, wavelet, level, threshold_method, threshold_factor,
                     denoise_strength, preserve_ratio, plot=False):
    """
    Process signal using Empirical Mode Decomposition + Wavelet.

    Parameters:
        y (array): Input signal
        n_noise_layers (int): Number of IMFs to denoise
        wavelet (str): Wavelet type
        level (int): Decomposition level
        threshold_method (str): Thresholding method
        threshold_factor (float): Threshold multiplier
        denoise_strength (float): Strength of denoising
        preserve_ratio (float): How much original signal to preserve
        plot (bool): Whether to plot IMFs

    Returns:
        tuple: (denoised_signal, noise_signal, imfs)
    """
    print("Starting EMD+Wavelet denoising process...")
    denoised_signal, noise_signal, imfs = perform_emd_wavelet_denoising(
        y, n_noise_layers, wavelet, level, threshold_method, threshold_factor,
        denoise_strength=denoise_strength, preserve_ratio=preserve_ratio
    )
    print("EMD+Wavelet denoising completed successfully")

    # Plot IMFs if requested
    if plot:
        print("Preparing IMF visualization...")

        # Create denoised IMFs for plotting
        denoised_imfs = imfs.copy()
        for i in range(min(n_noise_layers, imfs.shape[0])):
            denoised_imfs[i] = denoise_imf_with_wavelets(
                imfs[i], wavelet, level, threshold_method, threshold_factor
            )

        # Plot original and denoised IMFs
        plot_imfs(imfs, denoised_imfs, sr, n_noise_layers)

    return denoised_signal, noise_signal, imfs


def save_results_to_csv(csv_path, config_id, path_to_file, denoise_method, n_noise_layers,
                        wavelet, level, threshold_method, threshold_factor, denoise_strength,
                        preserve_ratio, sr, y, denoised_signal, noise_signal):
    """
    Save processing results to CSV file.

    Parameters:
        csv_path (str): Path to CSV file
        config_id (str): Unique configuration ID
        path_to_file (str): Path to audio file
        denoise_method (str): Denoising method used
        n_noise_layers (int): Number of layers denoised
        wavelet (str): Wavelet type used
        level (int): Decomposition level
        threshold_method (str): Thresholding method used
        threshold_factor (float): Threshold multiplier
        denoise_strength (float): Strength of denoising
        preserve_ratio (float): Ratio of original signal preserved
        sr (int): Sample rate
        y (array): Original signal
        denoised_signal (array): Denoised signal
        noise_signal (array): Extracted noise signal

    Returns:
        bool: Whether saving was successful
    """
    try:
        # Create a new dataframe with the results
        results = {
            'config_id': config_id,
            'file_path': path_to_file,
            'denoise_method': denoise_method,
            'n_noise_layers': n_noise_layers,
            'wavelet': wavelet,
            'level': level,
            'threshold_method': threshold_method,
            'threshold_factor': threshold_factor,
            'denoise_strength': denoise_strength,
            'preserve_ratio': preserve_ratio,
            'sr': sr,
            'original': y.tolist(),
            'denoised': denoised_signal.tolist(),
            'noise': noise_signal.tolist(),
            'processing_time': pd.Timestamp.now().isoformat()
        }

        # Check if the CSV file exists
        if os.path.exists(csv_path):
            # Load existing results
            existing_df = pd.read_csv(csv_path)

            # Check if this config already exists
            if config_id in existing_df['config_id'].values:
                # Update the existing row
                existing_df.loc[existing_df['config_id'] == config_id] = pd.Series(results)
            else:
                # Append new results
                existing_df = pd.concat([existing_df, pd.DataFrame([results])], ignore_index=True)

            # Save updated results
            existing_df.to_csv(csv_path, index=False)
            print(f"Updated existing results file: {csv_path}")
        else:
            # Create new results file
            pd.DataFrame([results]).to_csv(csv_path, index=False)
            print(f"Created new results file: {csv_path}")
        return True
    except Exception as e:
        print(f"Error saving results to CSV: {e}")
        return False


def normalize_signal_lengths(y, denoised_signal, noise_signal):
    """
    Ensure all signals have the same length.

    Parameters:
        y (array): Original signal
        denoised_signal (array): Denoised signal
        noise_signal (array): Noise signal

    Returns:
        tuple: (y, denoised_signal, noise_signal) - Normalized signals
    """
    min_length = min(len(y), len(denoised_signal), len(noise_signal))
    return y[:min_length], denoised_signal[:min_length], noise_signal[:min_length]


def wavelet_denoise(path_to_file, denoise_method='dwt', n_noise_layers=2, wavelet='db8', level=5,
                    threshold_method='soft', threshold_factor=0.9, plot=False,
                    denoise_strength=0.7, preserve_ratio=0.3, save_csv=False, csv_path=None):
    """
    Process audio file with wavelet transformation, remove noise, and plot results.
    Can save results to CSV and load from existing CSV if available.

    Parameters:
        path_to_file (str): Path to audio file
        denoise_method (str): Denoising method - 'dwt' for Discrete Wavelet Transform,
                              'wpt' for Wavelet Packet Transform, or 'emd' for
                              Empirical Mode Decomposition + Wavelet
        n_noise_layers (int): Number of detail coefficient layers to denoise (or IMFs for EMD)
        wavelet (str): Wavelet type to use
        level (int): Decomposition level (used for dwt and emd methods)
        threshold_method (str): 'soft' or 'hard' thresholding
        threshold_factor (float): Multiplier for threshold calculation (0.5-2.0 typical range)
        plot (bool): Whether to plot results
        denoise_strength (float): Strength of denoising from 0.0 (none) to 1.0 (full)
        preserve_ratio (float): How much of original signal to blend back in (0.0-1.0)
        save_csv (bool): Whether to save results to CSV
        csv_path (str): Path to save/load CSV file. If None, uses path_to_file with .csv extension

    Returns:
        tuple: (sr, y, denoised_signal, noise_signal) - Sample rate, original signal,
               denoised signal, and noise signal
    """
    # Generate configuration ID and CSV path
    if save_csv:
        config_id = generate_config_id(
            path_to_file, denoise_method, n_noise_layers, wavelet, level,
            threshold_method, threshold_factor, denoise_strength, preserve_ratio
        )
        csv_path = get_csv_path(path_to_file, csv_path, denoise_method)

        # Try to load existing results
        found, sr, y, denoised_signal, noise_signal = load_existing_results(csv_path, config_id)
        if found:
            # Plot signals if requested
            if plot:
                plot_signals(y, denoised_signal, noise_signal, sr, path_to_file, n_noise_layers)
            return sr, denoised_signal

    # Parse audio file if existing results weren't found
    sr, y = parse_file(path_to_file, save_csv)

    # Process using the requested method
    if denoise_method.lower() == 'dwt':
        denoised_signal, noise_signal = process_with_dwt(
            y, wavelet, level, n_noise_layers, threshold_method, threshold_factor, plot
        )
    elif denoise_method.lower() == 'wpt':
        denoised_signal, noise_signal = process_with_wpt(
            y, wavelet, n_noise_layers, threshold_method, threshold_factor
        )
    elif denoise_method.lower() == 'emd':
        denoised_signal, noise_signal, imfs = process_with_emd(
            y, n_noise_layers, wavelet, level, threshold_method, threshold_factor,
            denoise_strength, preserve_ratio, plot
        )
    else:
        raise ValueError(f"Unknown denoise method: {denoise_method}. Use 'dwt', 'wpt', or 'emd'.")

    # Normalize signal lengths
    y, denoised_signal, noise_signal = normalize_signal_lengths(y, denoised_signal, noise_signal)

    # Plot signals if requested
    if plot:
        plot_signals(y, denoised_signal, noise_signal, sr, path_to_file, n_noise_layers)

    # Save results to CSV if requested
    if save_csv:
        save_results_to_csv(
            csv_path, config_id, path_to_file, denoise_method, n_noise_layers,
            wavelet, level, threshold_method, threshold_factor, denoise_strength,
            preserve_ratio, sr, y, denoised_signal, noise_signal
        )

    return sr, denoised_signal
