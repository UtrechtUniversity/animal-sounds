# Synthetic Data

It proved difficult to collect enough Chimpanzee vocalizations in the jungle to train our models. But we had an abundance of vocalizations from the sanctuary. To increase and diversify our training set we have created synthetic samples by embedding the sanctuary vocalizations into the recorded background noise of the jungle.

## Table of Contents

- [How does it work](#how-does-it-work)
- [Software dependencies](#software-requirements)
- [Usage](#usage)
- [Remarks](#remarks)

## How does it work

This folder contains a number of scripts that form a pipeline to produce our synthetic samples. The pipeline consists of three steps:

1. Collect pure background fragments from our jungle recordings.
2. Create a time overview of all involved fragments, vocalizations and background noise, to ensure proper embedding. This step avoids embedding a long vocalization into a short noise fragment.
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

This folder contains three python scripts that form the pipeline, a shell script to run the entire pipeline and a folder `test_data` that contains all necessary data to demonstrate the pipeline. This folder consists of:

- A sub folder `recordings` in which we can find a small number of original jungle recordings.
- A text file `raven_annotations.txt` that contains timestamps 
- A sub folder `vocalizations` in which we 




: in the sub folder `recordings` one can find a small number of original jungle recordings, the `raven_annotations.txt` file contains timestamps in which we can find background noise of those recordings and in 


in the sub folder `vocalizations` we have put a small number WAV files of Chimpanzee vocalizations and the 