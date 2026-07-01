# Feature extraction
The modules in this directory are used to extract acoustic and/or deep learning features from '.wav' files. The features are used as input for the  [classifiers](../classifier), i.e. svm and cnn.

## Instructions

[Installation instructions](https://github.com/UtrechtUniversity/animal-sounds#getting-started)

## Audio Preprocessing

Raw audio recordings were loaded at a sampling rate of 48,000 Hz. A Butterworth bandpass filter was applied to isolate frequencies between 100 Hz and 2,000 Hz, corresponding to the primary frequency range of chimpanzee vocalizations while suppressing out-of-band environmental noise.

## Feature extraction for Support Vector Machines
We extract several feature sets from using:

- a [python version](https://github.com/mystlee/rasta_py) of the [rasta-mat](https://www.ee.columbia.edu/~dpwe/resources/matlab/rastamat/) library.
- an [Automatic Analysis Architecture](https://doi.org/10.5281/zenodo.1216028)

For our analyses we chunk all recordings into 0.5 second frames (with 0.25 second overlap between the chunks).
We apply a Butterworth bandpass filter for filtering audio between 100 and 2000 before extracting features. 
We create MFCC and RASTA-PLPC low level descriptors (LLDs) from the filtered signal. For each horizontal band of the MFCC and RASTA-PLPC representation we calculate $\Delta$ and $\Delta^2$, and extract statistical features from the plain LLDs, $\Delta$ and $\Delta^2$.

We extend the feature set with the features from an [Automatic Analysis Architecture](https://doi.org/10.5281/zenodo.1216028)

The script results in a feature set of 1140 features per audio frame.

### Running the script
Use shell script `run_svm.sh` to start `extract_features_svm.py` from the command line. The following arguments should be specified:
- `--config_file`; optional YAML file with reusable defaults and SVM extraction jobs.
- `--job_name`; optional job name under `feature_extraction_svm.jobs` in the config file.
- `--input_dir`; directory where the '.wav' files are located. This is typically defined in the selected config job.
- `--output_dir`; output path for the feature file ('.csv'). This is typically defined in the selected config job.
- `--frame_length`; subdivide '.wav' files in frames of this length (in number of samples, if the sample rate is 48000 samples per second, choose e.g. 24000 for 0.5 second frames)
- `--hop_length`; overlap between frames in number of samples per hop
- `--filter`; butter bandpass filter variables 

In `./config` the user can specify which features to extract and can define reusable SVM extraction defaults such as frame length, hop length, filter settings, number of cores, plus per-job input and output paths.

### sndfile library
If you get an error saying something about a 'snd_file' dependency on an ubuntu machine, this can be fixed by installing the following C library:
```
sudo apt-get install libsndfile-dev
```
## Feature extraction for Convolutional Neural Network (CNN)
 
A multi-channel feature representation was constructed for each audio recording using a four-stage pipeline developed by Librosa package[[1]](#ref1): mel spectrogram computation, Per-Channel Energy Normalization (PCEN)[[2]](#ref2), per-file z-normalization, and temporal derivative extraction.

### Mel Spectrogram

Mel-scaled spectrograms were computed from the filtered audio using a Hamming window with an FFT window length of 750 samples and a hop length of 376 samples, yielding 64 mel-frequency bins. The spectrogram was computed in linear (power) scale to serve as input for PCEN.

| <img src="../../img/melspectrogram.png" width="400" /> |

### Per-Channel Energy Normalization (PCEN)

Rather than applying conventional logarithmic compression, we applied PCEN (Wang et al., 2017) to the mel spectrogram. PCEN performs adaptive gain control that suppresses stationary background noise while enhancing transient acoustic events such as vocalizations. The PCEN parameters were set as follows: gain = 0.98, bias = 2, power = 0.5, time constant = 1.5, and epsilon = 1e-6. PCEN was applied to the full-length recording before any temporal segmentation to ensure that the automatic gain control had sufficient temporal context for accurate noise floor estimation.

### Per-File Z-Normalization

To reduce environment-specific variation in absolute energy levels, per-file z-normalization was applied to each PCEN-normalized spectrogram. For each frequency bin, the mean and standard deviation were computed across the time axis of the full recording, and the spectrogram was normalized to zero mean and unit variance:

$$\hat{S}(f, t) = \frac{S(f, t) - \mu_f}{\sigma_f + \epsilon}$$

where $S(f, t)$ is the spectrogram value at frequency bin $f$ and time frame $t$, $\mu_f$ and $\sigma_f$ are the mean and standard deviation across time for frequency bin $f$, and $\epsilon = 10^{-8}$ prevents division by zero. This normalization ensures that the model learns relative spectral patterns (e.g., a vocalization standing out from its local background) rather than absolute energy levels that differ across recording environments and equipment.

### Delta Features

First- and second-order temporal derivatives delta and delta-delta[[3]](#ref3) were computed from the normalized spectrogram using a Savitzky-Golay-style filter with a width of 9 frames. These derivatives capture the temporal dynamics of the spectral content — onset and offset patterns (delta) and their rate of change (delta-delta). The original spectrogram and its two derivatives were stacked to form a three-channel input representation of shape (3, 64, T), where T is the number of time frames in the recording. For recordings shorter than the minimum required filter length, edge-padding was applied before computing derivatives and the result was trimmed to the original length.

### Running the script
Open a command line and run the following command:
```
sh run_feature_extraction_dl.sh
```

This command applies `extract_features_dl.py` on the whole dataset. The following arguments should be specified:
- `--input_dir`; directory where the '.wav' files are located.
- `--output_dir`; directory where the feature files ('.pkl') should be stored.
- `--label`; the label of the wav file, i.e. chimpanze or background
- `--config_file`; the path to the configuration file that contains the following parameter values:
   - `sample_rate`; sample rate to read the .wav files
   - `window_length`; subdivide '.wav' files in frames of this length (in number of samples, in our case, the sample rate is 48000 samples per second, we chose 750 for 15-millisecond frames)
   - `hop_length`; overlap between frames in number of samples per hop (in our case, the sample rate is 48000 samples per second, we chose 376)
   - `n_mel`; number of mel features, i.e. horizontal bars in spectrogram, which in our case it is 64.
   - `low_cut`; minimum frequency for butter bandpass filter
   - `high_cut`; maximum frequency for butter bandpass filter
   - `pcen_params`; list of pcen parameters and their values

## References

<a id="ref1">1. </a>  McFee, B., et al. (2015). librosa: Audio and music signal analysis in Python. Proceedings of the 14th Python in Science Conference.

<a id="ref2">2. </a> Wang, Y., et al. (2017). Trainable frontend for robust and far-field keyword spotting. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).

<a id="ref3">3. </a>  Furui, S. (1986). Speaker-independent isolated word recognition using dynamic features of speech spectrum. IEEE Transactions on Acoustics, Speech, and Signal Processing, 34(1), 52–59.


