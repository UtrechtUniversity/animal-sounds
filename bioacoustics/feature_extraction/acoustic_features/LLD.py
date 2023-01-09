"""Module for feature extraction from audio files."""

import aiofiles
import asyncio
import librosa
import numpy as np
import pandas as pd

from . import speech_features as sf

from .tools import butter_bandpass_filter, extract_features


# Low Level Descriptors
class LLD:
    """Extracting MFCC and RASTA-PLP features from audio files.

    Parameters
    ----------
    file_set : list
        List of audio files.
    frame_length : int
        Length of the FFT window.
    hop_length : int
        Number of samples between successive frames.
    sr : int
        Sampling rate of audio files.
    bandpass_filter : tuple, optional
        Tuple of low and high cut frequencies for bandpass filter.
    config : dict, optional
        Configuration dictionary.
    features : list, optional
        List of features to extract.
    """

    def __init__(
        self,
        file_set,
        frame_length,
        hop_length,
        sr,
        bandpass_filter=False,
        config=False,
        features=[],
    ):
        self.file_set = file_set
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.sr = sr
        self.bandpass_filter = None

        # This should be another argument
        self.dtype = np.int16
        self.norm_factor = 1.0
        # I don't understand this, but if we are dealing with
        # integers we have to divide by the minimum signed int value
        if np.issubdtype(self.dtype, np.integer):
            self.norm_factor = np.abs(np.iinfo(self.dtype).min)

        # bandpass filter, either a tuple or False
        if (isinstance(bandpass_filter, tuple)) and len(bandpass_filter) == 3:
            self.bandpass_filter = tuple([int(i) for i in bandpass_filter])
        elif (isinstance(bandpass_filter, tuple)) and len(bandpass_filter) == 0:
            self.bandpass_filter = False
        else:
            raise RuntimeError("bandpass_filter is either False or tuple")

        self.config = config
        self.features = features

    async def extract(self):
        """Schedule feature extraction tasks and return results.

        Returns
        -------
        pd.DataFrame
            Dataframe with extracted features.
        """

        result = []
        for chunk in self.file_set:
            tasks = []
            for file in chunk:
                if file is not None:
                    t = asyncio.create_task(self.__extract_features(file))
                    tasks.append(t)
            result += [await task for task in tasks]
        result = [r for r in result if r is not False]
        return pd.concat(result)

    async def __open_wav_file(self, path):
        """Open wav file and return signal.

        Parameters
        ----------
        path : str
            Path to audio file.

        Returns
        -------
        np.ndarray
            Array of audio signal.
        """
        async with aiofiles.open(path, mode="br") as f:
            contents = await f.read()
            f.close()
            # first 22 bytes are not relevant
            signal = np.frombuffer(contents, self.dtype)[22:]
            # librosa divides by min-value of int dtype
            signal = signal / self.norm_factor
            return signal

    # extract speech features
    def extract_speech_features(self, frames):
        """Extract speech features from frames.

        Parameters
        ----------
        frames : np.ndarray
            Array of frames.

        Returns
        -------
        np.ndarray
            Array of speech features.
        """

        return np.apply_along_axis(sf.extract_speech_features, 1, frames)

    async def __extract_features(self, path):
        """Extract features from audio file.

        Parameters
        ----------
        path : str
            Path to audio file.

        Returns
        -------
        pd.DataFrame
            Dataframe with extracted features.
        """
        # filepath
        filepath = str(path)

        # this is the async part: wait for the IO reading
        signal = await self.__open_wav_file(filepath)

        # run bandpass filter if necessary
        if self.bandpass_filter:
            org_dtype = signal.dtype
            signal = butter_bandpass_filter(
                signal,
                self.bandpass_filter[0],
                self.bandpass_filter[1],
                self.sr,
                self.bandpass_filter[2],
            )
            # make sure we are still dealing with float32 values
            signal = signal.astype(org_dtype)

        # padding of signal with 0's if it is too short
        if len(signal) < self.frame_length:
            return False

        # create frames
        f = librosa.util.frame(
            signal, frame_length=self.frame_length, hop_length=self.hop_length
        )

        frames_no = np.shape(f)[1]
        # create meta data
        meta_data = [[filepath, x] for x in range(frames_no)]

        # transpose the frames table: rows = #frames, cols = samples
        learningData = f.T

        # extract the features
        learningFeatures = extract_features(
            self.config, learningData, self.features, self.sr, meta_data
        )

        # # Scale features and store scaler
        # scaler = preprocessing.StandardScaler().fit(learningFeatures)
        # learningFeatures = scaler.transform(learningFeatures)

        # part II, speech features
        speech_features = pd.DataFrame(list(self.extract_speech_features(learningData)))

        # put into DataFrame
        df_features = pd.DataFrame(learningFeatures, columns=self.features.featuresRef)
        df_meta_data = pd.DataFrame(meta_data, columns=["file_path", "frameId"])
        df_meta_data["length[s]"] = learningData.shape[1] / self.sr
        df_res = pd.concat([df_meta_data, df_features, speech_features], axis=1)

        return df_res
