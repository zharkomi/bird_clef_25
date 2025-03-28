import numpy as np


def trim_silence(audio, sr=44100, frame_ms=20, hop_ms=10, threshold_factor=0.05, min_silence_ms=100):
    """
    Adaptively trim silence/noise from the beginning and end of an audio signal.

    Parameters:
    -----------
    audio : numpy.ndarray
        Input audio signal (1D array)
    sr : int, optional
        Sample rate of the audio signal (default: 44100)
    frame_ms : int, optional
        Frame length in milliseconds (default: 20)
    hop_ms : int, optional
        Hop length in milliseconds (default: 10)
    threshold_factor : float, optional
        Factor to determine energy threshold relative to max energy (default: 0.05)
        Lower values are more aggressive in keeping audio
    min_silence_ms : int, optional
        Minimum silence duration in ms to consider for trimming (default: 100)

    Returns:
    --------
    numpy.ndarray
        Trimmed audio signal with silence removed from beginning and end
    """
    # Convert ms parameters to samples
    frame_length = int(sr * frame_ms / 1000)
    hop_length = int(sr * hop_ms / 1000)
    min_silence = int(sr * min_silence_ms / 1000)

    # Ensure audio is 1D
    audio = audio.flatten()

    # Calculate energy in frames
    energy = []
    frames = []

    for i in range(0, len(audio) - frame_length, hop_length):
        frame = audio[i:i + frame_length]
        frames.append((i, i + frame_length))
        energy.append(np.sum(frame ** 2) / frame_length)

    energy = np.array(energy)

    # Calculate adaptive threshold based on signal characteristics
    # Use percentile instead of max to be robust against outliers
    energy_threshold = threshold_factor * np.percentile(energy, 95)

    # Find frames above threshold
    above_threshold = energy > energy_threshold

    # If no frames above threshold, return empty array
    if not np.any(above_threshold):
        return np.array([])

    # Find first and last frame above threshold
    first_frame = np.where(above_threshold)[0][0]
    last_frame = np.where(above_threshold)[0][-1]

    # Convert back to samples
    start_sample = frames[first_frame][0]
    end_sample = frames[last_frame][1]

    # Add a buffer to avoid cutting off the beginning or end of meaningful signal
    buffer_samples = int(sr * 50 / 1000)  # 50ms buffer

    start_sample = max(0, start_sample - buffer_samples)
    end_sample = min(len(audio), end_sample + buffer_samples)

    return audio[start_sample:end_sample]


def trim_silence_adaptive(audio, sr=44100, frame_ms=20, hop_ms=10, min_silence_ms=100):
    """
    More sophisticated adaptive silence trimming using noise floor estimation.
    Better for varying noise conditions.

    Parameters:
    -----------
    Same as trim_silence, but without threshold_factor as it's determined adaptively

    Returns:
    --------
    numpy.ndarray
        Trimmed audio signal with silence removed from beginning and end
    """
    # Convert ms parameters to samples
    frame_length = int(sr * frame_ms / 1000)
    hop_length = int(sr * hop_ms / 1000)
    min_silence = int(sr * min_silence_ms / 1000)

    # Ensure audio is 1D
    audio = audio.flatten()

    # Calculate energy in frames
    energy = []
    frames = []

    for i in range(0, len(audio) - frame_length, hop_length):
        frame = audio[i:i + frame_length]
        frames.append((i, i + frame_length))
        energy.append(np.sum(frame ** 2) / frame_length)

    energy = np.array(energy)

    # Estimate noise floor using the first and last 10% of frames
    # Assuming the start and end are more likely to contain silence
    frames_10pct = max(5, len(energy) // 10)
    potential_noise = np.concatenate([energy[:frames_10pct], energy[-frames_10pct:]])

    # Calculate noise statistics
    noise_mean = np.mean(potential_noise)
    noise_std = np.std(potential_noise)

    # Set adaptive threshold based on noise statistics
    # 3 standard deviations above mean covers 99.7% of noise
    adaptive_threshold = noise_mean + 3 * noise_std

    # Find frames above threshold
    above_threshold = energy > adaptive_threshold

    # If no frames above threshold, try a more aggressive approach
    if not np.any(above_threshold):
        # Fall back to a percentile-based approach
        adaptive_threshold = np.percentile(energy, 75)
        above_threshold = energy > adaptive_threshold

        # If still no frames, return the original audio
        if not np.any(above_threshold):
            return audio

    # Find first and last frame above threshold
    first_frame = np.where(above_threshold)[0][0]
    last_frame = np.where(above_threshold)[0][-1]

    # Convert back to samples
    start_sample = frames[first_frame][0]
    end_sample = frames[last_frame][1]

    # Add a buffer to avoid cutting off the beginning or end of meaningful signal
    buffer_samples = int(sr * 50 / 1000)  # 50ms buffer

    start_sample = max(0, start_sample - buffer_samples)
    end_sample = min(len(audio), end_sample + buffer_samples)

    return audio[start_sample:end_sample]
