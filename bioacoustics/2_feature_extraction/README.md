# Feature extraction

The codes in this folder are used to extract acoustic and/or deep learning features from '.wav' files. The features are used as input for the classifier ([step 3](../3_classifier)).

## Instructions

[Installation instructions](https://github.com/UtrechtUniversity/animal-sounds/tree/documenation_svm#getting-started)

### Feature extraction for Support Vector Machines
We extract several feature sets from using:
- a [python version](https://github.com/mystlee/rasta_py) of the [rasta-mat](https://www.ee.columbia.edu/~dpwe/resources/matlab/rastamat/) library.
- an [Automatic Analysis Architecture](https://doi.org/10.5281/zenodo.1216028)

For our analyses we chunk all recordings into 0.5 second frames (with 0.25 second overlap between the chunks).
We apply a Butterworth bandpass filter for filtering audio between 100 and 2000.

The script results in a feature set of 1140 features per audio frame.

#### Running the script
Use shell script `run.sh` to start `main.py` from the command line. The following arguments should be specified:
- `--input_dir`; directory where the '.wav' files are located.
- `--output_dir`; directory where the feature files ('.csv') should be stored.
- `--frame_length`; subdivide '.wav' files in frames of this length (in number of samples, if the sample rate is 48000 samples per second, choose e.g. 24000 for 0.5 second frames)
- `--hop_length`; overlap between frames in number of samples per hop
- `--filter`; butter bandpass filter variables 

In `./config` the user can specify which features to extract.

## sndfile library
If you get an error saying something about a 'snd_file' dependency on an ubuntu machine, this can be fixed by installing the following C library:
```
sudo apt-get install libsndfile-dev
```
