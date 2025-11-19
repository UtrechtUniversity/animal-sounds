# 🐍 Bioacoustics Project — Setup & Processing Guide

This guide walks you through downloading the software, setting up Python, organizing your data and running the pipeline for audio preprocessing, feature extraction and classification. 

> [!IMPORTANT]
> Assumption: You have a terminal (aka command line interface). If you are working on a Mac or Linux, you should have an appropriate terminal installed. If you are working on Windows, we recommend installing [Git Bash](https://gitforwindows.org/). If you have no prior experience with the command line, we recommend spending a couple of hours practicing with [navigating files and folders in your terminal](https://utrechtuniversity.github.io/workshop-introduction-to-bash/modules/module2.html).

## ⚙️ 1. Setup instructions

### 1.1 Clone the repository

Open a terminal (e.g. Git Bash if you are working on Windows), navigate to the directory where you want to store this project and run the following command:

```bash
git clone https://github.com/UtrechtUniversity/animal-sounds.git
```

### 1.2.1 Install `uv`

Install **uv**, a fast package and environment manager for Python.  
📖 [Installation guide](https://docs.astral.sh/uv/getting-started/installation/)

### 1.2.2 Create a Virtual Environment

Assuming you have cloned this repository (step 1.1), navigate into the cloned directory on your terminal (`cd animal-sounds`) and run the following command:

```bash
uv sync
```
### 1.2.3 Activate the environment and install the project

Run the following command:

```bash
source .venv/bin/activate
uv pip install -e .
```

## 🎧 2. Creating Audio Segments from raw recordings and annotations

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

### 🗂️ Step 0: Prepare Input Data

✅ **Annotations**:  
Ensure that annotation files are in the correct **format**:

The pipeline expects a text file with output from Raven. Within Raven Pro the user can select which columns should be printed in the `.txt` file. The columns that are needed are:

```| begin path | end path | class | file offset (s) | start time (s) | end time (s) |
---- | --------- | ------ | ---------- | ------| ---- | 
```

✅ **Data Organization:**  
Use the `data` folder to organize your data in the following structure: 

```
animal-sounds/
├── data/
│   ├── original_recordings/
│   ├── processed_wav_files/
│   │   ├── vocalizations/
│   │   └── background/
│   ├── annotation_txt_files/
│   │   ├── vocalizations/
│   │   └── background/
│   ├── condensed_wav_files/
│   ├── synthetic_intermediate/
│   ├── features/
│   └── synth_data/
|     ├── vocalizations/
|     └── background/
└── ...
```

🛠️ **Configure Script Paths**

> [!IMPORTANT]
> If you are organizing your data in a different way, please make sure to adapt the `.sh` scripts that are used below to point to the correct folders.

### Step 1: Create audio segments for annotated segments

This step is done using the `raven_to_wav.sh` shell script. The purpose of this step is to cut out audio segments from the original recordings that are annotated to contain a particular sound (e.g. a chimp vocalization, or background sound).

Make sure you are in the folder called `animal-sounds` in the terminal and run the following command:

Run:
```bash
./bioacoustics/wav_processing/raven_to_wav/raven_to_wav.sh 
```
If all went correctly, you should now have `.wav` files in the `processed_wav_files/vocalizations` folder. If you are planning to run this script for multiple species, please organize the output into folders for each species.

### Step 2:Condensation (optional)

This step is done using the `extract_chimps.sh` shell script. The purpose of this step is to capture audio segments from the original recordings that show an increase in energy and are therefore more likely to contain a chimp vocalization. This condensed audio still needs to be annotated by a human, but it is expected to be faster than using the original recordings.

Run
```bash
./bioacoustics/wav_processing/condensation/extract_chimps.sh
```

> [!IMPORTANT]
> This step is optional. If you are not planning to run this step, make sure to read the extended instructions [here](https://github.com/UtrechtUniversity/animal-sounds/tree/main/bioacoustics/wav_processing/condensation), some initial annotations are needed to tune the parameters for better results.


### Step 3: Synthetic data

This step is done using the `synth_pipeline.sh` shell script. The purpose of this step is to create synthetic data by combining the audio segments created in the 2.1 and background sounds. The script doesn't use all the segments at once, but takes a random sample of 30 files as input. To create more synthetic data, simply run the script another time. The resulting data is saved in the `synth_data` folder. 

Run
`./bioacoustics/wav_processing/synthetic_data/synth_pipeline.sh`

## 3. Feature extraction

The purpose of this step is to transform the audio segments into features (a set of numerical values) that can be used for training an SVM classifier. We are combining several feature extraction methods in this step. The resulting features are saved in the `features` folder.

Run

```bash
./bioacoustics/feature_extraction/run_feature_extraction_svm.sh
```
