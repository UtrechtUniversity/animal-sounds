# Raven to wav

To reduce the volume of data needed for training the classifiers we extract annotated parts from the original `.wav` files and store them separately.

## Table of Contents

- [How does it work](#how-does-it-work)
- [Software dependencies](#software-requirements)
- [Usage](#basic-usage)
    - [Shell script](#shell-script)
- [Remarks](#remarks)

## How does it work

`raven_to_wav.py` parses the `.txt` output from Raven Pro software. Within Raven Pro the user can select which columns should be printed in the `.txt` file.
The columns that are needed to run `raven_to_wav.py` are:
| begin path | end path | class | file offset (s) | start time (s) | end time (s) |
 ---- | --------- | ------ | ---------- | ------| ---- | 

The script contains a function to account for annotations that start in one file and end in another file. It splits those annotations into 2 or more `.wav` files.
The output filenames of the `.wav` files are numbered and contain metadata (recorder ID, original wav file, and start time(AKA offset)).

## Software requirements

- Python 3
- [Librosa ~0.9.1](https://librosa.org/doc/latest/index.html)
- [Pandas ~1.3.4](https://pandas.pydata.org)
- [Soundfile ~0.10.3](https://pysoundfile.readthedocs.io/en/latest/)

## Basic usage

This folder contains a python script to run process the original `.wav` files and a `.sh` shell script to call the python script.
The python script can be run from command line, in Pycharm (modify run configuration to pass the arguments) or using the shell script.

```
python3.8 raven_to_wav.py --annotations_file="raven_annotations.txt"
		--species=Chimpanzee 
		--wavpath=/data/original_wav_files/ 
		--outputdir=/data/processed_wav_files/ 
		--recID=2C 
		--min_sig_len=0.2 
		--bg_padding_len=0.05 
		--createframes=0

```

### Shell script
A convenient shell script that runs the script can be found in this folder as well, and can be executed with:
```
$ ./raven_to_wav.sh 
```

## Remarks
