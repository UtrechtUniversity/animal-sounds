import librosa
import numpy as np
import scipy.io.wavfile as wv
from scipy.signal import stft


class Extractor:

    # init: read wav file, or set to empty signal.
    def __init__(self, wav_file_path=None):
        self.path = wav_file_path
        # for short term fourier transform
        self.stft = None

        # load wav file, or initialize with empty numpy array
        if wav_file_path:
            self.sr, self.signal = self.load_wav_file(self.path)
        else:
            self.signal, self.sr = np.array([]), None

    # set signal manually
    def set_signal(self, signal, sr):
        self.signal, self.sr = signal, sr

    # + operator add signals if sr is identical
    def __add__(self, other):
        if self.sr is not None and self.sr == other.sr:
            new_signal = np.concatenate((self.signal, other.signal))
            new_filter = Extractor()
            new_filter.set_signal(new_signal, self.sr)
            return new_filter
        else:
            raise Exception("You tried to add two signals with different sample rates.")

    # length of signal
    def __len__(self):
        return self.signal.shape[0]

    # minimal vocalization duration in seconds
    # padding in seconds (before and after vocalization)
    # min/max_threshold_db: loudness of vocalizations I am looking for, above ref db's
    def detect_vocalizations(
        self,
        freqs=None,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        min_threshold_db=30,
        max_threshold_db=None,
        threshold_pattern=0.1,
        ref=np.median,
        min_voc_duration=0.5,
        padding=0.3,
        ignore_voc=0.0,
        use_cached_stft=True,
    ):

        if not use_cached_stft or (use_cached_stft and self.stft is None):
            # perform stft
            self.stft = stft(
                self.signal,
                fs=self.sr,
                nfft=n_fft,  # nfft: powers of 2
                nperseg=win_length,
                noverlap=(win_length - hop_length),
            )

        # get stft output
        f, T, x_stft = self.stft

        # convert to dbs
        x_stft = np.nan_to_num(x_stft)
        power = np.abs(x_stft)
        # take power relative to median
        dbs = librosa.power_to_db(power**2, ref=ref)

        # default values for frequencies of interest
        if freqs is None:
            freqs = [(0, max(f))]

        # collect all time indexes in which something happens
        time_indexes = set([])

        # get all indexes of dbs rows of every band that we're
        # interested in
        for (low, high) in freqs:
            idx_low = (np.abs(f - low)).argmin() - 1
            idx_low = 0 if idx_low < 0 else idx_low
            idx_high = (np.abs(f - high)).argmin() + 1
            idx_high = len(f) - 1 if idx_high >= len(f) else idx_high

            # list of all frequencies within current band
            band = list(range(idx_low, idx_high + 1))

            # apply frequency indexes to dbs
            dbs_filtered = dbs[band, :]

            # STEP 1: collect loud noises
            # find time indexes where we can find a noise above
            # <threhold_db> on top of median IN the relevant frequency bands
            max_dbs = np.max(dbs_filtered, axis=0)

            # print(np.median(max_dbs))

            # set all values below (and above) threshold_db to 0
            if max_threshold_db is None:
                max_dbs[max_dbs < min_threshold_db] = 0
            else:
                max_dbs[(max_dbs < min_threshold_db) | (max_dbs > max_threshold_db)] = 0
            # collect time indexes where noise is above threshold
            time_indexes_dbs = set(np.nonzero(max_dbs)[0])
            # add to our indexes
            time_indexes = time_indexes.union(time_indexes_dbs)

            # STEP 2: collect noises that stand out
            # based on cumulative freq distributions
            # first create a relevant frequency range
            dbs_filtered = dbs[list(range(min(band), max(band) + 1)), :]
            dbs_suppressed = self._suppress_noise_patterns(dbs_filtered, window=5)
            # sum the noise suppressed over the frequency axis
            deviations = np.sum(dbs_suppressed, axis=0)
            # normalize: biggest value is 1
            deviations = deviations / np.max(deviations)
            # filter anything below <threshold_pattern> of max signal
            # by default 10%
            deviations[deviations < threshold_pattern] = 0
            # and collect all time indices which stick out
            time_indexes_patterns = set(np.nonzero(deviations)[0])
            # add to our indexes
            time_indexes = time_indexes.union(time_indexes_patterns)

        # these are the time indexes of interest collected from analyzing
        # the individual frequency bands
        time_indexes = sorted(list(time_indexes))

        # I need sensible clusters for the found time indexes:
        # divide total time in bins and create histograms of time indexes.
        # merge adjacent bins when they contain more than 1 value. Bins with
        # 0 indexes mark the end of a cluster
        # divide the duration of the wav file in bins of size
        # <min_voc_duration> to cluster the time-indexes
        number_of_bins = int(T[-1] / min_voc_duration)
        bins = np.linspace(0, len(T), num=number_of_bins)
        clusters, _ = np.histogram(time_indexes, bins=bins)

        # form the clusters, adjacent bins with more than 1 value in them
        # will be merged.
        clustered_indexes = []
        cluster = []
        for i, cluster_size in enumerate(clusters):
            if cluster_size > 0:
                cluster += time_indexes[0:cluster_size]
                time_indexes = time_indexes[cluster_size:]
                # append the last cluster to cluster_indexes
                if i == len(clusters) - 1:
                    clustered_indexes.append(cluster)
            else:
                # add cluster to clustered_indexes
                if len(cluster) > 0:
                    clustered_indexes.append(cluster)
                    cluster = []

        # at this stage I have an array with clustered indexes, now I want
        # to have start and end times of these clusters
        clusters = []
        for c in clustered_indexes:
            start = T[c[0]]
            end = T[c[-1]]
            if (end - start) > ignore_voc:
                start = max(start - padding, 0)
                end = min(end + padding, T[-1])
                clusters.append((start, end))

        return clusters

    # padding in seconds
    def extract_intervals(self, timestamps=[], padding=0, dtype=np.int16):
        cut_signal = np.array([])
        padding_zeros = np.zeros(int(padding * self.sr))

        for start, end in timestamps:
            start_signal_index = self._time_2_sample_unit(start, "start")
            end_signal_index = self._time_2_sample_unit(end, "end")
            cut_signal = np.concatenate(
                (
                    cut_signal,
                    self.signal[start_signal_index:end_signal_index],
                    padding_zeros,
                )
            )

        cut_signal = np.asarray(cut_signal, dtype=dtype)
        tmp = Extractor()
        tmp.set_signal(cut_signal, sr=self.sr)
        return tmp

    def _time_2_sample_unit(self, t, time_type):
        return (
            int(np.floor(t * self.sr))
            if time_type == "start"
            else int(np.ceil(t * self.sr))
        )

    def _create_freq_distribs(self, dbs, window_y=4):
        all_distribs = {}

        for i in range(dbs.shape[0] - window_y):
            # take a small frequency band over time (all columns)
            # and sort all db values found in this band
            # reshape to a single array
            values = np.sort(dbs[i: i + window_y, :].reshape(-1))
            # # set all negative values to 0 (power lower than median)
            # values[values < 0] = 0
            # create bins for these sorted values
            bins = np.linspace(np.min(values) - 20, np.max(values) + 20, num=600)
            # create a histogram from bins and values
            hist, _ = np.histogram(values, bins=bins)
            # create a normalized cum distribution out of this
            cum_dist = np.cumsum(hist) / np.cumsum(hist)[-1]
            # and get the bin in which we find the highest frequency
            highest_bin = np.where(cum_dist == 1)[0][0]
            # for every frequency band, store the cumulative power distribution
            # up to the highest bin and the used bins
            all_distribs[i] = (cum_dist[:highest_bin], bins[:highest_bin])

        return all_distribs

    def _compute_window(
        self, dbs, x_start, y_start, distribs, alpha=3, window_x=4, window_y=4
    ):
        # collect all db values
        x_sample = np.sort(
            dbs[y_start: y_start + window_y, x_start: x_start + window_x].reshape(-1)
        )
        sample_cum = (1 + np.arange(len(x_sample))) / len(x_sample)
        # and arrange a similar cumulative power/db distribution
        # as we have created for the frequency band of y_start in
        # _create_freq_distribs. The latter is taken over the
        # entire time interval. The one we are creating right now is taken
        # over a short amount of time (window_y and window_x)
        cum_dist, bins = distribs[y_start]
        #
        inter_sample = np.interp(bins, x_sample, sample_cum, left=0, right=1)
        # return the sum of all difference between both cum. distributions
        return np.sum((cum_dist - inter_sample) ** alpha)

    def _suppress_noise_patterns(self, dbs, window=4):
        all_distrib = self._create_freq_distribs(dbs, window_y=window)

        signal = np.zeros(dbs.shape)
        for i_x in range(dbs.shape[1] - window):
            for i_y in range(dbs.shape[0] - window):
                new_signal = self._compute_window(dbs, i_x, i_y, all_distrib)
                # new_signal can be negative, take the max of new_signal and 0
                # leaving us with only positive differences. So if things are louder
                # in specific time/frequency window in comparison with the cum
                # power distribution of the entire time interval we get a positive number
                # back.
                signal[i_y, i_x] = max(new_signal, 0)
        return signal

    # load wav file
    def load_wav_file(self, path: str):
        return wv.read(path)

    # write wav file
    def to_wav(self, path):
        wv.write(path, self.sr, self.signal)
