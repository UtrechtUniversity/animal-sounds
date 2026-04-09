"""Extracting features of audio files using melspectrogram."""

import glob
import librosa
import logging
import pandas as pd
import numpy as np
import argparse
import os
from acoustic_features.tools import butter_bandpass_filter
import yaml


def parse_arguments():
    # parse arguments if available
    parser = argparse.ArgumentParser(description="Deep learning features")
    parser.add_argument(
        "--config_file", type=str, help="File path to the config file"
    )
    parser.add_argument(
        "--input_dir", type=str, help="File path to the dataset of .wav files"
    )

    parser.add_argument("--output_dir", type=str, default=None, help="output dir")
    parser.add_argument("--label", type=str, help='label : "chimpanze" or "background"')
    return parser


def extract_features(
    fp, sample_rate, window_length, hop_length, n_mel,
    low_cut, high_cut, pcen_params
):
    """Load audio from .wav file, filter it, and pass it to function
    compute_melspectrogram_with_fixed_length().

    Parameters
    ----------
    fp: str
        audio file_path
    sample_rate: int
        sampling rate of fp
    window_length: int
        length of the FFT window
    hop_length: int
        number of samples between successive frames
    n_mel: int
        number of mel features, i.e. horizontal bars in spectrogram
    new_img_size : list of int
        the target size of the image e.g. 64 64
    low_cut: int
        minimum frequency
    high_cut: int
        maximum frequency

    Returns
    -------
    np.ndarray:
        Mel spectrogram.
    """

    y, sr = librosa.load(fp, sr=sample_rate)
    y_filtered = butter_bandpass_filter(y, low_cut, high_cut, sr)
    melspectrogram_db = compute_melspectrogram_with_fixed_size(
        y_filtered, sample_rate, window_length, hop_length, n_mel,
        pcen_params
    )
    return melspectrogram_db


def compute_melspectrogram_with_fixed_size(
    audio, sample_rate, window_length_set, hop_length_set, n_mel_set,
        pcen_params=None, normalize=True, use_deltas=True
):
    """Create PCEN-normalized melspectrogram for a given audio.
    PCEN (Per-Channel Energy Normalization) suppresses stationary background
    noise and enhances transient events, making features environment-invariant.

    Parameters
    ----------
    audio: np.ndarray
        audio time-series.
    sample_rate: int
        sampling rate of fp
    window_length_set: list
        length of the FFT windows, a list of three
    hop_length_set: list
        number of samples between successive frames, a list of three
    n_mel_set: int
        number of mel features, i.e. horizontal bars in spectrogram
    new_img_size: list
        the target size of the images

    Returns
    -------
    np.ndarray:
        PCEN-normalized Mel spectrogram.
    """

    if pcen_params is None:
        pcen_params = {
            "gain": 0.98,
            "bias": 2,
            "power": 0.5,
            "time_constant": 0.4,
            "eps": 1e-6,
        }
    try:
        specs = []
        num_channels = len(window_length_set)

        for i in range(num_channels):
            window_length = window_length_set[i]
            hop_length = hop_length_set[i]
            n_mel = n_mel_set[i]

            # compute a mel-scaled spectrogram (linear scale, NOT log)
            # PCEN requires linear scale input, not log-compressed
            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio,
                sr=sample_rate,
                hop_length=hop_length,
                n_fft=window_length,
                n_mels=n_mel,
                window="hamming",
            )

            # apply PCEN instead of log compression
            # hop_length must match the one used above
            spec = librosa.pcen(
                mel_spectrogram * (2**31),
                sr=sample_rate,
                hop_length=hop_length,
                **pcen_params,

            )
            spec = spec.astype(np.float32)

            if normalize:
                spec = per_file_znorm(spec)
            if use_deltas:
                spec = add_deltas(spec)  # (n_mels, n_frames) -> (3, n_mels, n_frames)

            specs.append(spec)

    except Exception as e:
        print("\nError encountered while parsing files\n>>", e)
        return None

    return specs


def per_file_znorm(spec, eps=1e-8):
    """
    Per-file z-normalization across the time axis for each frequency bin.

    Removes environment-specific absolute energy levels while preserving
    the relative spectral shape of transient events (e.g., chimp calls).

    Parameters
    ----------
    spec : np.ndarray
        Spectrogram of shape (n_mels, n_frames).
    axis : int
        Axis along which to compute stats. Default -1 (time).
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    np.ndarray
        Z-normalized spectrogram, same shape as input.
    """
    mean = spec.mean(axis=1, keepdims=True)
    std = spec.std(axis=1, keepdims=True) + eps
    return (spec - mean) / std


def add_deltas(spec, width=9):
    """
    Add delta and delta2 channels to a spectrogram.

    Parameters
    ----------
    spec : np.ndarray
        Shape (n_mels, n_frames) — single channel spectrogram.
    width : int
        Filter width for delta computation. Default 9.

    Returns
    -------
    np.ndarray
        Shape (3, n_mels, n_frames).
    """

    n_frames = spec.shape[-1]

    # librosa.feature.delta needs at least (width) frames
    # for very short files, pad temporarily, compute, then trim
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


def get_label(lbl):
    return lbl


if __name__ == "__main__":

    parser = parse_arguments()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    sample_rate = config["feature_extraction"]["sample_rate"]
    window_length = config["feature_extraction"]["window_length"]
    hop_length = config["feature_extraction"]["hop_length"]
    n_mel = config["feature_extraction"]["n_mel"]
    low_cut = config["feature_extraction"]["low_cut"]
    high_cut = config["feature_extraction"]["high_cut"]
    pcen_params = config["feature_extraction"].get("pcen", None)

    input_dir = args.input_dir
    output_dir = args.output_dir
    label = args.label

    out_dir = os.path.dirname(output_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    all_files = glob.glob(input_dir)
    logging.info("input_dir: %s", input_dir)
    logging.info("number of selected files: %d", len(all_files))

    df_features = pd.DataFrame(columns=["file_path", "features", "label_1"])
    for f in all_files:

        melspectrogram = extract_features(
            f,
            sample_rate,
            window_length,
            hop_length,
            n_mel,
            low_cut,
            high_cut,
            pcen_params=pcen_params,
        )

        label_1 = get_label(label)
        new_df = pd.DataFrame(
            {"file_path": [f], "features": [melspectrogram], "label_1": [label_1]}
        )
        logging.info("processing file %s", f)
        df_features = pd.concat([df_features, new_df], join="inner").copy()

    if df_features is not None:
        df_features.to_pickle(output_dir)
        logging.info("df_features.shape %s", df_features.shape)
