# Using the trained models on your own data
If you have a dataset that is highly similar as the data that is used for training our models, you could consider applying the trained model for classifying your data. Similar would mean: Classifying chimpanzees in a tropical rainforest with similar background noise and using the same type of recorders.


## Obtain the repository
The commands below can be run on a Unix terminal. It assumes you have [python3.8](https://www.python.org/) installed.
On windows, preferably use `git bash` or similar software to run the commands below.

First navigate to the directory where you want to store this project.
Next, obtain all scripts and methods in this repository clone the repository as follows:

```
git clone https://github.com/UtrechtUniversity/animal-sounds.git
```
## Install requirements

Install all required python libraries:
```
cd animal-sounds
python -m pip install -r requirements.txt
```
## Organize data folder
- Create a data folder and subfolders to store your data following the project structure below.
- Move or copy your audio files to the subfolders.
- Create a features folder in the output folder (see project structure below)
```
.
├── .gitignore
├── CITATION.md
├── LICENSE.md
├── README.md
├── requirements.txt
├── bioacoustics              <- main folder for all source code
│   ├── 1_wav_processing 
│   ├── 2_feature_extraction
│   └── 3_classifier        
├── data               <- All project data, ignored by git
│   ├── original_wav_files
│   ├── processed_wav_files            
│   └── txt_annotations           
└── output
    ├── features        <- Figures for the manuscript or reports, ignored by git
    ├── models          <- Models and relevant training outputs
    ├── notebooks       <- Notebooks for analysing results
    └── results         <- Graphs and tables

```

## Preprocess data
### SVM: Configure and run feature extraction scripts
- Go to the feature extraction folder
```
cd bioacoustics/2_feature_extraction
```
- Run `main.py` from the command line as follows, but change the options `--input_dir`, `--output_dir` and `--cores` (number of CPU cores to run the task on) when applicable:
```
python main.py --input_dir "../../data/processed_wav_files/" --output_dir "../../output/features/features.csv" --frame_length 24000 --hop_length 12000 --cores 2 --filter 100 2000 5 --label1 unknown --label2 unknown
```

- Alternatively, use the text file `run.sh` to customize your runs of `main.py`.  the variable `input_dir=<location of your files>` so it will point to the location of your files and (if necessary) `output_dir=<location of your files>` so it will point to the location where you want your feature files to be stored. Change the variable `--cores <number of available CPU cores>` to the number of available CPU cores on your machine. Potentially you will have to 
add permission to execute the file.

If all went correct your output will look like this:
```
Number of processors on your machine:  2
Running on 2 cores.
Read 9 files in 0.0003058910369873047 sec
Processed in 13.261967658996582 sec
```
If you see this error:
```
OSError: sndfile library not found
```
You have to install the following C library:
```
sudo apt-get install libsndfile-dev
```
### DL: Create chunks

## Predict
### Classification using SVM 
In the previous step, `.wav` files are chunked into 0.5 second chunks. These chunks are then translated into features describing the chunk.
In this step we classify each chunk into classes `chimpanze` or `background`. 

- Go to the directory `3_classifier`
`cd ../3_classifier`

- Use this command to run the predict script (but customize the options to choose the model you wish to use):
```
python predict.py --model=svm --feature_dir=../../output/features/ --trained_model_path=../../output/models/svm/all/svm_model.sav --output_dir=../../output/models/svm/all/predictions/
```


