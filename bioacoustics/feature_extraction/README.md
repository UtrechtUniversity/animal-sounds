# Feature Extraction Module

This README is now a routing page. The canonical feature extraction documentation lives on the Quarto site.

Use these pages instead:

- [Feature extraction module documentation](https://utrechtuniversity.github.io/animal-sounds/feature-extraction.html)

This folder still contains the extraction code for both SVM and CNN pipelines, along with the shell scripts and configuration used by those workflows.
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


