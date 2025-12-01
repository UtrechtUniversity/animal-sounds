# 🐍 Predict with SVM

There are two situations in which you would want to predict:

- You have an annotated dataset that you want to use for evaluating the performance of your trained model
- You want to apply the model to new data to find vocalizations

This tutorial will cover the first case for now, but might be extended in the future for the second case.

## 🎧 2. Creating Audio Segments from raw recordings and annotations

This is a three-step process to create audio segments for 
The relevant scripts is stored in `bioacoustics/wav_processing/raven_to_wav`. 

### 🗂️ Step 0: Prepare Input Data

✅ **Annotations**:  
Ensure that annotation files are in the correct **format**:

The pipeline expects a text file with output from Raven. Within Raven Pro the user can select which columns should be printed in the `.txt` file. The columns that are needed are:

```
| begin path | end path | class | file offset (s) | start time (s) | end time (s) |
---- | --------- | ------ | ---------- | ------| ---- | 
```

✅ **Data Organization:**  
Use the `predict_data` folder to organize your data in the following structure: 

```
animal-sounds/
├── predict_data/
│   ├── original_recordings/
│   ├── processed_wav_files/
│   │   ├── vocalizations/
│   │   └── background/
│   ├── annotation_txt_files/
│   │   ├── vocalizations/
│   │   └── background/
│   ├── features/
│   └── predictions/
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
./bioacoustics/wav_processing/raven_to_wav/raven_to_wav.sh predict_data
```
If all went correctly, you should now have `.wav` files in the `processed_wav_files/vocalizations` folder. If you are planning to run this script for multiple species, please organize the output into folders for each species.

## 3. Feature extraction

The purpose of this step is to transform the audio segments into features (a set of numerical values) that can be used for precition using the previously trained SVM classifier. We are combining several feature extraction methods in this step. The resulting features are saved in the `features` folder.

Run

```bash
./bioacoustics/feature_extraction/run_feature_extraction_svm.sh predict_data
```
If all went correctly, you should now have `.csv` files in the `features` folder. This step can take hours to complete, but the script is parallelized, so running it on a (virtual) machine with many cores is recommended. The `--cores` parameter can be used to specify the number of cores to use, by default it uses all cores - 2.

## 4. Training

The purpose of this step is apply the previously trained SVM classifier on the features created in the previous step. The resulting predictions are stored in the `predictions` folder. The Unweighted Average Recall (UAR) is used as a performance metric and is displayed in the terminal.

Run

```bash
./bioacoustics/classifier/predict_svm.sh
```
If all went correctly, you should see the UAR the terminal. 

