# Condensation

To speed up the annotation process of the large amount of recordings we have created a script that tries to detect and gather audio fragments that stand out with respect to the background noise of the jungle. These fragments _might_ contain Chimpanzee vocalizations. By applying this script we were able to reduce the annotation workload: instead of listening to many hours of recordings we could present a drastically 'condensed' version of the data.

## Table of Contents

- [How does it work](#how-does-it-work)
- [Software dependencies](#software-requirements)
- [Usage](#usage)
    - [Using the Extractor class](#using-the-extractor-class)
    - [Using the extractor script](#using-the-extractor-script)
    - [Using the shell script](#using-the-shell-script)
- [Remarks](#remarks)

## How does it work

The detection of deviating audio patterns comprises a Short-time Fourier transform to produce a power distribution in the time-frequency domain. Depending on the properties of the expected primate vocalizations we discard redundant frequency bands. From the remaining bands we collect small time-intervals in which the registered signal loudness exceeds a species specific threshold, or in which the local cumulative power distribution deviates from a global counterpart.
This collection represents a set of timestamps where we expect to hear disruptions in the ambient noise. The time-intervals are used to extract the corresponding signal fragments from our raw data. These fragments are bundled into a new audio file which a resulting high density of vocalizations that can be annotated more efficiently.

## Software dependencies

- Python 3
- [Librosa ~0.9.1](https://librosa.org/doc/latest/index.html)
- [Numpy ~1.20.3](https://numpy.org/)
- [Scipy ~1.8.0](https://scipy.org/)
- [Soundfile ~0.10.3](https://pysoundfile.readthedocs.io/en/latest/)

## Usage

This folder contains 2 Python files, a shell script and a folder containing one WAV file for testing:

* The `extractor.py` module contains a class that analyses WAV fi`les as described above.
* The `condensate.py` script applies the Extractor class and facilitates batch processing.
* The shell script, `extract_chimps.sh` was used to run the processing of multiple batches.
* The `test_data` folder contains a WAV file for demonstration purposes.

### Using the Extractor class

The script below demonstrates the basic usage of the Extractor class:
```
from extractor import Extractor

# Create an Extractor instance
data_in = Extractor('./test_data/20191220_190302.WAV')

# Perform the detection, detect anomalies between
# 200 and 1000 kHz, with a volume between 20 to 30 dB
# above the median volume.
# The result, timestamps, will be a list containing
# timestamps of fragments that deviate.
# Note that multiple frequency bands can be analyzed.
timestamps = data_in.detect_vocalizations(
    freqs=[(200, 1000)],
    min_threshold_db=20, 
    max_threshold_db=30
)

# Extract the anomalies from the wav file into a 
# new Extractor instance, pad fragments with 0.1 secs
# of silence.
extracted = data_in.extract_intervals(
    timestamps, 
    padding=0.1
)

# Save the collection of fragments as a new WAV file
extracted.to_wav('out.wav')
```

### Using the extractor script

Use the `condensate.py` script to process a large amount of input data. The script takes a single WAV file or folder containing multiple WAV files as input argument. All important detection parameters from the Extractor class are available in this script as well. It produces a WAV file containing _all_ detected anomalies and a CSV file that provides an overview where (file) and when (timestamp) these anomalies were found. Usage:

```
$ python condensate.py --input="./test_data" \
    --output-csv="timestamps.csv" \
    --output-signal="out.wav" \
    --frequencies="[(200, 1000)]" \
    --volume="(20, 30)"
```

### Using the shell script

The `extract_chimps.sh` automates running the `condensate.py` file. Convenient if you need to condensate data in multiple folders. Usage is trivial:

```
$ ./extract_chimps.sh
```

## Remarks

Reducing the annotation time with the Extractor class requires some manual labor! Some experimentation was needed to find the correct frequency band in which the majority of Chimpanzee vocalizations could be found. We also had to deal with differences in background noise per recording batch, which forced us to use different volume thresholds in some cases. Note that this is _not_ a smart detection algorithm. It is not designed to detect vocalizations produced by animals. The sound of an airplane flying by, and the 'bang' of a closing door will be selected as well for annotation. Despite the disability to filter out only animal vocalizations we could remove a significant, uneventful portion from our entire dataset.
