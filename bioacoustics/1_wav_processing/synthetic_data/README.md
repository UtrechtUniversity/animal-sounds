# Synthetic Data

It proved difficult to collect enough Chimpanzee vocalizations in the jungle to train our models. But we had an abundance of vocalizations from the sanctuary. To increase and diversify our training set we have created synthetic samples by embedding the sanctuary vocalizations into the recorded background noise of the jungle.

## Table of Contents

- [How does it work](#how-does-it-work)
- [Software dependencies](#software-requirements)
- [Usage](#basic-usage)
    - [Step 1](#step-1)
    - [Step 2](#step-2)
    - [Step 3](#step-3)
    - [Shell script](#shell-script)
- [Remarks](#remarks)

## How does it work

This folder contains a number of scripts that form a pipeline to produce our synthetic samples. The pipeline consists of three steps:

1. Collect pure background noise fragments from our jungle recordings.
2. In this step we register filenames and their duration (in the form of time-frames) of the background noise fragments and the vocalizations. The generated overviews ensure fast and proper embedding.
3. Produce the synthetic data based on the overviews generated in the previous step. Randomly embed vocalizations into suitable noise fragments. Every vocalization produces multiple samples in which the vocalization's presence varies in loudness.

## Software requirements

- Python 3
- [Librosa ~0.9.1](https://librosa.org/doc/latest/index.html)
- [Numpy ~1.20.3](https://numpy.org/)
- [Pandas ~1.3.4](https://pandas.pydata.org)
- [Pydub ~0.25.1](https://pypi.org/project/pydub/)
- [Scipy ~1.8.0](https://scipy.org/)
- [Soundfile ~0.10.3](https://pysoundfile.readthedocs.io/en/latest/)

## Basic usage

This folder contains the three numbered python scripts that form the pipeline, a shell script to run the entire pipeline and a sub folder `test_data` that stores all necessary data to demonstrate the pipeline. This sub folder consists of:

- A folder `recordings` in which we can find a small number of original jungle recordings.
- A text file `raven_annotations.txt` that contains timestamps of jungle noise fragments found in the WAV files of the `recordings` folder.
- A folder `vocalizations` in which a small number of Chimpanzee vocalizations are stored.

### Step 1

The first script, `1_collect_background.py` reads any number of Raven annotation files (with a `.txt` extension) and parses their contents, collecting the paths of the recordings involved and the relevant timestamps of fragments that were annotated as 'Background'. It then extracts the timestamps from these files and stores them in a folder of choice. To execute the script on our test data, please run:
```
$ python 1_collect_background.py \
    --input_dir './test_data/recordings/' \
    --annotation_dir './test_data' \
    --output_dir './test_data/results/background'
```
where `--input_dir` denotes the folder that contains the jungle recordings, `--annotation_dir` denotes the folder with the Raven annotation files and `--output_dir` denotes the folder in which the scripts collects the background noise fragments.

### Step 2

The second script, `2_create_overview.py` takes an input set of WAV files and registers their absolute filepaths and their duration. It produces a json file containing these registrations. In the next step we need such overviews for the set of background noise fragments and vocalizations so we can mix and match properly. Provided you have executed the previous step on the test data, you can run this script like this:
```
# create an overview for the background fragments
$ python 2_create_overview.py \
    --input_dir './test_data/results/background' \
    --output './test_data/results/overviews/background.json'

# create an overview for the vocalizations
$ python 2_create_overview.py \
    --input_dir './test_data/vocalizations' \
    --output './test_data/results/overviews/vocalizations.json'
```

### Step 3

In the third and final step, performed by the `3_create_synth_sample.py` script, we mix the vocalizations into the background fragments. Per vocalization a suitable candidate from the background collection is randomly selected. In case the algorithm can't find a suitable candidate because the duration of the vocalization is too long, then the vocalization is chopped up into smaller fragments and the selection is repeated for the fragments. Per vocalization 4 new versions are created in which the loudness of the vocalization is increasingly dampened. Every version gets its own numerical suffix, denoting the amount of dampening in dB, multiplied by 10. If we continue our test example, we run this script like this:
```
$ python 3_create_synth_sample.py \
    --primate_json './test_data/results/overviews/vocalizations.json' \
    --background_json './test_data/results/overviews/vocalizations.json' \
    --output './test_data/results/synth_data/'
```

### Shell script
A convenient shell script that runs all the demonstration steps consecutively can be found in this folder and can be executed with:
```
$ ./synth_pipeline.sh 
```

## Remarks


