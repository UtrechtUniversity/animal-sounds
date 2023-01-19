# Chunk wav

Chunk_wav component splits `.wav` files into smaller chunks. This is a step in the preparation process for deep 
learning models.

## Table of Contents

- [How does it work](#how-does-it-work)
- [Software dependencies](#software-requirements)
- [Usage](#basic-usage)
    - [Shell script](#shell-script)
- [Remarks](#remarks)

## How does it work
Here you can find the inputs to `make_chunks.py`:
*  Directory of wav files
*  Desired length of wav files in seconds 
*  Overlap of two following chunks in seconds
*  Directory of output wav files

By default, desired length of wav files is 0.5 seconds. There is an overlap of 0.25 seconds between each of two 
following chunks.

## Software requirements

- Python 3
- [numpy ~1.23.2](https://numpy.org)
- [scipy~1.9.1](https://scipy.org)

## Basic usage

This folder contains a python script to process the original `.wav` files and a `.sh` shell script to call the python script.
The python script can be run from command line.

```
python make_chunks.py 
        --input_dir
        --chunk_length 
        --overlap
        --output_dir
```

### Shell script
Here is a convenient shell script that runs the python script for the entire dataset.
```
$ ./split_dataset.sh 
```

## Remarks
