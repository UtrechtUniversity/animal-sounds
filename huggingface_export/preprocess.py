"""
Standalone feature extraction for chimp-vocalization-cnn10 / cnn12 models.

Reproduces the exact preprocessing pipeline used at training time:
    raw audio -> Butterworth bandpass filter -> mel spectrogram (linear scale)
    -> PCEN normalization -> per-file z-normalization -> delta + delta-delta channels

Output shape: (3, n_mel, n_frames) -- ready to be chunked into 0.5s frames
and passed to CNN10Hub / CNN12Hub.

Dependencies: librosa, numpy, scipy
"""

import numpy as np
import librosa
from scipy.signal import butter, filtfilt

# ---- Exact config used to train the released checkpoints ----
SAMPLE_RATE = 48000
WINDOW_LENGTH = 750
HOP_LENGTH = 376
N_MEL = 64
LOW_CUT = 100
HIGH_CUT = 2000
PCEN_PARAMS = {
    "gain": 0.98,
    "bias": 2,
    "power": 0.5,
    "time_constant": 1.5,
    "eps": 1e-6,
}
FRAMES_PER_CHUNK = 64  # 0.5s chunks at this hop/sample rate


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    if not lowcut or not highcut:
        return data
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)


def per_file_znorm(spec, eps=1e-8):
    """Per-file z-normalization across the time axis for each frequency bin."""
    mean = spec.mean(axis=1, keepdims=True)
    std = spec.std(axis=1, keepdims=True) + eps
    return (spec - mean) / std


def add_deltas(spec, width=9):
    """Stack spec with delta and delta-delta -> shape (3, n_mel, n_frames)."""
    n_frames = spec.shape[-1]
    min_frames = width + 1
    if n_frames < min_frames:
        pad_amount = min_frames - n_frames
        spec_padded = np.pad(spec, ((0, 0), (0, pad_amount)), mode="edge")
        delta = librosa.feature.delta(spec_padded, order=1, width=width)[:, :n_frames]
        delta2 = librosa.feature.delta(spec_padded, order=2, width=width)[:, :n_frames]
    else:
        delta = librosa.feature.delta(spec, order=1, width=width)
        delta2 = librosa.feature.delta(spec, order=2, width=width)
    return np.stack([spec, delta, delta2], axis=0)


def extract_features(file_path):
    """
    Load a .wav file and compute the 3-channel PCEN mel spectrogram
    expected by CNN10Hub / CNN12Hub.

    Parameters
    ----------
    file_path : str
        Path to a .wav audio file.

    Returns
    -------
    np.ndarray, shape (3, n_mel, n_frames)
        Feature array. Split into (3, n_mel, 64)-frame chunks before
        feeding to the model (see `chunk_features` below).
    """
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    y_filtered = butter_bandpass_filter(y, LOW_CUT, HIGH_CUT, sr)

    mel_spectrogram = librosa.feature.melspectrogram(
        y=y_filtered,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        n_fft=WINDOW_LENGTH,
        n_mels=N_MEL,
        window="hamming",
    )

    # PCEN requires linear-scale mel input, not log-compressed
    spec = librosa.pcen(
        mel_spectrogram * (2**31),
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        **PCEN_PARAMS,
    ).astype(np.float32)

    spec = per_file_znorm(spec)
    spec = add_deltas(spec)  # (n_mel, n_frames) -> (3, n_mel, n_frames)

    return spec


def chunk_features(spec, frames_per_chunk=FRAMES_PER_CHUNK):
    """
    Split a (3, n_mel, n_frames) feature array into non-overlapping
    (3, n_mel, frames_per_chunk) chunks, padding the final chunk if needed.

    Returns
    -------
    np.ndarray, shape (n_chunks, 3, n_mel, frames_per_chunk)
    """
    n_frames = spec.shape[-1]
    n_full_chunks = n_frames // frames_per_chunk
    remainder = n_frames % frames_per_chunk

    chunks = []
    for c in range(n_full_chunks):
        start = c * frames_per_chunk
        chunks.append(spec[:, :, start:start + frames_per_chunk])

    if remainder > 0:
        start = n_full_chunks * frames_per_chunk
        last_chunk = spec[:, :, start:]
        pad_width = frames_per_chunk - last_chunk.shape[-1]
        last_chunk = np.pad(last_chunk, ((0, 0), (0, 0), (0, pad_width)), mode="constant")
        chunks.append(last_chunk)

    return np.stack(chunks, axis=0)


if __name__ == "__main__":
    import sys
    import torch

    if len(sys.argv) != 2:
        print("Usage: python preprocess.py <path_to_wav_file>")
        sys.exit(1)

    spec = extract_features(sys.argv[1])
    chunks = chunk_features(spec)
    print(f"Extracted features: {spec.shape} -> {chunks.shape[0]} chunk(s) of shape {chunks.shape[1:]}")

    # Example: ready to feed into CNN10Hub / CNN12Hub
    # from modeling import CNN10Hub
    # model = CNN10Hub.from_pretrained("utrechtuniversity/chimp-vocalization-cnn10-synthetic")
    # model.eval()
    # with torch.no_grad():
    #     probs = torch.softmax(model(torch.from_numpy(chunks).float()), dim=1)
    # print(probs)
