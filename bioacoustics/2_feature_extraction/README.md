# Feature extraction

The codes in this folder are used to extract acoustic and/or deep learning features from '.wav' files. The features are used as input for the classifier ([step 3](../3_classifier)).

## Instructions

### SVM
Use shell script `run.sh` to start `main.py` from the command line. The following arguments should be specified:
- `--input_dir`; directory where the '.wav' files are located.
- `--output_dir`; directory where the feature files ('.csv') should be stored.
- `--frame_length`; subdivide '.wav' files in frames of this length (in number of samples, if the sample rate is 48000 samples per second, choose e.g. 24000 for 0.5 second frames)
- `--hop_length`; overlap between frames in number of samples per hop
- '--filter'; butter bandpass filter variables 

In ./config the user can specify which features to extract.


## Installation

The codes are developed for Python 3.8
The following packages can installed e.g. using pip.

### Python packages

- pandas
- librosa
- sympy
- aiofiles
- spectrum

## sndfile library
If you get an error saying something about a 'snd_file' dependency on an ubuntu machine, this can be fixed as follows:
```
sudo apt-get install libsndfile-dev
```
