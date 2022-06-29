# Synthetic Data

It proved difficult to collect enough Chimpanzee vocalizations in the jungle to train our models. But we had an abundance of vocalizations from the sanctuary. To increase and diversify our training set we have created synthetic samples by embedding the sanctuary vocalizations into the recorded background noise of the jungle.

## Table of Contents

- [How does it work](#how-does-it-work)
- [Software dependencies](#software-requirements)
- [Usage](#usage)
    - [The pipeline](#the-pipeline)
- [Remarks](#remarks)

## How does it work

This folder contains a number of scripts that form a pipeline to produce our synthetic samples. The pipeline consists of three steps:

1. Collect pure background fragments from our jungle recordings.
2. Create a time overview of all involved fragments, vocalizations _and_ background noise, to ensure proper embedding. This step avoids embedding a long vocalization into a short noise fragment.
3. Produce the synthetic data. Randomly embed vocalizations into suitable noise fragments. Every vocalization produces multiple samples in which the vocalization's presence varies.

## Software requirements

- Python 3
- [Librosa ~0.9.1](https://librosa.org/doc/latest/index.html)
- [Numpy ~1.20.3](https://numpy.org/)
- [Pandas ~1.3.4](https://pandas.pydata.org)
- [Pydub ~0.25.1](https://pypi.org/project/pydub/)
- [Scipy ~1.8.0](https://scipy.org/)
- [Soundfile ~0.10.3](https://pysoundfile.readthedocs.io/en/latest/)

## Usage

This folder contains the three numbered python scripts that form the pipeline, a shell script to run the entire pipeline and a sub folder `test_data` that stores all necessary data to demonstrate the pipeline. This sub folder consists of:

- A folder `recordings` in which we can find a small number of original jungle recordings.
- A text file `raven_annotations.txt` that contains timestamps of jungle noise fragments found in the WAV files of the `recordings` folder.
- A folder `vocalizations` in which a small number of Chimpanzee vocalizations are stored.

### The pipeline

The first script, `1_collect_background.py` reads any number of Raven annotation files (with a `.txt` extension) and parses their contents, collecting the paths of the recordings involved and the relevant timestamps of fragments that were annotated as 'Background'. It then extracts the timestamps from these files and stores them in a folder of choice. To execute the script on our test data, please run:

```
$ python 1_collect_background.py \
    --input_dir './test_data/recordings/' \
    --annotation_dir './test_data' \
    --output_dir './test_data/results'
```
where `--input_dir` denotes the folder that contains the jungle recordings, `--annotation_dir` denotes the folder with the Raven annotation files and `--output_dir` denotes the folder in which the scripts collects the background noise fragments.