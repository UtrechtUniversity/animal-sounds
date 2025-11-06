

# 🐍 Bioacoustics Project — Setup & Processing Guide

This guide walks you through setting up your development environment and running the audio preprocessing pipelines for annotated, condensed, and synthetic data.


---

## ⚙️ 1. Environment Setup

Follow these steps to get your Python environment ready.

### 1.1.1 Install `uv`

Install **uv**, a fast package and environment manager for Python.  
📖 [Installation guide](https://github.com/astral-sh/uv)

### 1.1.2 Create a Virtual Environment

Assuming you have [cloned this repository](README.md), navigate to the cloned directory on your laptop and run the following command:

```bash
uv sync
```
### 1.1.3 Activate the Environment

Run the following command:

```bash
source .venv/bin/activate
uv pip install -e .
```

## 🎧 Creating Audio Segments from raw recordings and annotations

This is a three-step process to create audio segments for training a classifier.
The relevant scripts are stored in `bioacoustics/wav_processing/`. The directory structure is displayed below, where the files ending with `.sh` are shell scripts that can be executed from the command line to run all three steps. The directories contains python scripts (`.py`) and some other files that are not in this overview:

```
bioacoustics/
├── wav_processing/
│   ├── raven_to_wav/
│   │   └── raven_to_wav.sh
│   ├── condensation/
│   │   └── extract_chimps.sh
│   └── synthetic_data/
│       └── synth_pipeline.sh
└── ...
```

### 🗂️ 1.2 Prepare Input Data

- ✅ **Annotations**:  
  Ensure that annotation files are:
  - in the correct **format**  
  - located in the correct **directory**  
  > ⚠️ *TODO: Specify annotation format and directory path.*

- ✅ **Recordings**:  
  Verify that all recordings referenced by the annotations are located in the correct directory.  
  > ⚠️ *TODO: Specify recordings directory.*

### 🛠️ 1.3 Configure Script Paths

If you have created new folders to organize your data in the previous step, ensure all all shell scripts in `bioacoustics/wav_processing/` are configured to use the correct paths to the folders you have created.


## 2. Create audio segments for annotated segments

Make sure you are in the folder called `animal-sounds` and run the following command:

Run:
```bash
./bioacoustics/wav_processing/raven_to_wav/raven_to_wav.sh
```
> ⚠️ *TODO: add success message to script*

If all went correctly, you should now have `.wav` files in the `processed_wav_files/vocalizations` folder.

## Condensation (optional)

make sure recordings to apply this on are in the right directory

Adapt shell script path names in: `bioacoustics/wav_processing/condensation/extract_chimps.sh`


Run
`./bioacoustics/wav_processing/condensation/extract_chimps.sh`

## Synthetic data

Adapt shell script path names in `/bioacoustics/wav_processing/synthetic_data/synth_pipeline.sh`

Run
`./bioacoustics/wav_processing/synthetic_data/synth_pipeline.sh`
