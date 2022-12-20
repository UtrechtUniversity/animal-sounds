"""Script to extract features of .wav files using melspectrogram."""

import glob
import librosa
import pandas as pd
import numpy as np
import argparse
import os
import sys
from acoustic_features.tools import butter_bandpass_filter
from PIL import Image


def parse_arguments():
    # parse arguments if available
    parser = argparse.ArgumentParser(
        description="Deep learning features"
    )

    # File path to the data.
    parser.add_argument(
        "--input_dir",
        type=str,
        help="File path to the dataset of .wav files"
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='output dir'
    )

    parser.add_argument(
        '--label',
        type=str,
        help='label : "chimpanze" or "background"'
    )
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=48000,
        help='sample rate'
    )
    parser.add_argument(
        '--window_length',
        type=int,
        nargs="+",
        #default= 750 1504 6000
        help='length of the FFT window'
    )

    parser.add_argument(
        '--hop_length',
        type=int,
        nargs="+",
        #default= 376 752 3000
        help='number of samples between successive frames'
    )

    parser.add_argument(
        '--n_mel',
        type=int,
        nargs="+",
        # default=64, for resnet: 64 32 8
        help='number of mel features, i.e. horizontal bars in spectrogram'
    )

    parser.add_argument(
        '--new_img_size',
        type=int,
        nargs="+",
        # default= 64 64, for resnet 224 224
        help='the target size of the images'
    )

    parser.add_argument(
        '--low_cut',
        type=int,
        default=100,
        help='minimum desired frequency'
    )
    parser.add_argument(
        '--high_cut',
        type=int,
        default=2000,
        help='maximum desired frequency'
    )
    return parser


def extract_features(fp, sample_rate, window_length, hop_length, n_mel, new_img_size,
                     low_cut, high_cut):
    """Load audio from .wav file, filter it, and pass it to compute_melspectrogram_with_fixed_length() function .

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
    y, sr = librosa.load(fp, sr=args.sample_rate)
    y_filtered = butter_bandpass_filter(y, low_cut, high_cut, sr)
    melspectrogram_db = compute_melspectrogram_with_fixed_size(y_filtered, sample_rate, window_length,
                                                               hop_length, n_mel, new_img_size)
    return melspectrogram_db


def compute_melspectrogram_with_fixed_size(audio, sample_rate, window_length_set, hop_length_set, n_mel_set,
                                           new_img_size):
    """Create melspectrogram for a given audio

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
                Mel spectrogram.
    """
    try:
        specs = []
        num_channels = len(window_length_set)
        img_width = new_img_size[0]
        img_height = new_img_size[1]
        for i in range(num_channels):
            window_length = window_length_set[i]
            hop_length = hop_length_set[i]
            n_mel = n_mel_set[i]

            # compute a mel-scaled spectrogram
            # https://github.com/kamalesh0406/Audio-Classification/blob/master/preprocessing/preprocessingESC.py
            mel_spectrogram = librosa.feature.melspectrogram(y=audio,
                                                             sr=sample_rate,
                                                             hop_length=hop_length,
                                                             n_fft=window_length,
                                                             n_mels=n_mel,
                                                             window='hamming')

            eps = 1e-6
            spec = np.log(mel_spectrogram + eps)

            if spec.shape[1] != img_height:
                spec = np.array(Image.fromarray(spec).resize((img_width, img_height)))

            specs.append(spec)

        print(len(specs))
    except Exception as e:
        print("\nError encountered while parsing files\n>>", e)
        return None

    return specs


def get_label(lbl):
    return lbl


if __name__ == '__main__':

    # get arguments
    parser = parse_arguments()
    args = parser.parse_args()


    out_dir = os.path.dirname(args.output_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    all_files = glob.glob(args.input_dir)
    print('number of selected files', len(all_files))

    df_features = pd.DataFrame(columns = ['file_path','features', 'label_1'])
    for f in all_files:

        melspectrogram = extract_features(f, args.sample_rate, args.window_length, args.hop_length, args.n_mel,
                                          args.new_img_size, args.low_cut, args.high_cut)

        label_1 = get_label(args.label)
        new_df = pd.DataFrame({'file_path': [f], 'features': [melspectrogram], 'label_1': [label_1]})
        print('processing file ', f)
        df_features = pd.concat([df_features,new_df], join='inner').copy()

    print(df_features['file_path'])
    if df_features is not None:
        df_features.to_pickle(args.output_dir)
        print('df_features.shape', df_features.shape)