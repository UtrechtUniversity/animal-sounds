"""Script to make .wav files of the same length."""
import os
import glob
import argparse
import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import write


def parse_arguments():
    """Parse arguments if available."""
    parser = argparse.ArgumentParser(description="Bioacoustics")
    parser.add_argument("--input_dir", type=str, help="Directory of wav files")
    parser.add_argument(
        "--chunk_length",
        type=float,
        default=0.5,
        help="Desired length of wav files in seconds ",
    )

    parser.add_argument(
        "--overlap",
        type=float,
        default=0.25,
        help="overlap of two following chunks in seconds",
    )
    parser.add_argument("--output_dir", type=str, help="Directory of output wav files")
    return parser


def make_audio_chunks(audio, sample_rate, chunk_length, overlap):
    """Split an audio with a specific chunk length and an overlap.

    Parameters
    ----------
    audio: np.ndarrary
        audio time series.
    sample_rate: int
        sample rate of the given audio
    chunk_length: float
        length of desired chunks in second
    overlap: float
        the overlap of two chunks in second

    Returns
    -------
        a list of chunked signals
    """
    slices = np.arange(
        0, len(audio) / sample_rate, chunk_length - overlap
    )  # , dtype=np.int64
    audio_slice = []
    for start, end in zip(slices[:-1], slices[1:]):
        start_audio = start * sample_rate
        end_audio = (end + overlap) * sample_rate
        # print(start_audio, end_audio)
        audio_slice.append(audio[int(start_audio): int(end_audio)])

    return audio_slice


def pad_audio(audio, sample_rate, target_length, output_fp, save_output=True):
    """Pad an audio with digital silence to the specified duration.

    Parameters
    ----------
    audio: np.ndarrary
        audio time series.
    sample_rate: int
        sample rate of the given audio
    target_length: int
        desired length of wav files in seconds
    output_fp: str
        file path of output wav file
    save_output: bool
        Indicates if output should be saved

    Returns
    -------
    np.ndarrary:
        Padded audio file.
    """
    # Calculate target number of samples
    n_tar = int(sample_rate * target_length)
    # Create the target shape
    n_pad = n_tar - len(audio)
    padded = np.hstack((np.zeros(n_pad, dtype=np.int8), audio))

    if save_output:
        new_file_name = output_fp + "_1.wav"
        write(new_file_name, sample_rate, padded)
    return padded


def split_audio(audio, sample_rate, chunk_length, overlap, output_fp):
    """Split an audio to a number of chunks with the specified duration .

    Parameters
    ----------
    audio: np.ndarrary
        audio time series.
    sample_rate: int
        sample rate of audio
    chunk_length: float
        length of the desired chunks in second
    overlap: float
        overlap of two following chunks in seconds
    output_fp: str
        file path of output wav file
    """
    chunks = make_audio_chunks(
        audio, sample_rate, chunk_length, overlap
    )  # Make chunks with size of chunk_length
    if len(chunks[-1]) / sample_rate < chunk_length:
        chunks[-1] = pad_audio(
            chunks[-1], sample_rate, chunk_length, output_fp, save_output=False
        )

    # Export all of the individual chunks as wav files
    for i, chunk in enumerate(chunks):
        chunk_name = output_fp + "_chunk{0}.wav".format(i)
        print("exporting", chunk_name)
        write(chunk_name, sample_rate, chunk)


def main():
    """Split wav files in to a specific length with overlap."""
    parser = parse_arguments()
    args = parser.parse_args()

    chunk_length = args.chunk_length
    overlap = args.overlap
    fps = glob.glob(args.input_dir)

    print("number of selected files", len(fps))
    file_no = 0
    for fp in fps:
        file_no += 1
        print(fp)
        print(file_no)

        # read audio file
        sample_rate, audio = wavfile.read(fp)

        # get file name
        base = os.path.basename(fp)
        file_name = os.path.splitext(base)[0]
        out_file_path = args.output_dir + "/" + file_name

        if len(audio) / sample_rate < chunk_length:
            pad_audio(audio, sample_rate, chunk_length, out_file_path, save_output=True)
        else:
            split_audio(audio, sample_rate, chunk_length, overlap, out_file_path)


if __name__ == "__main__":
    main()
