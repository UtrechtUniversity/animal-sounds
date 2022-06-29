# animal-sounds

<!-- Include Github badges here (optional) -->
<!-- e.g. Github Actions workflow status -->

The aim of this software is to classify Chimpanze vocalizations in audio recordings from the tropical rainforests of Africa. The software can be used for processing raw audio data, extracting features, and apply and compare Support Vector Machines and Deep learning methods for classification. The pipeline is reusable for other settings and species or vocalization types as long as a certain amount of labeled data has been collected. The best performing models will be available here for general usage.

<!-- TABLE OF CONTENTS -->
## Table of Contents

- [Animal Sounds](#animal-sounds)
  - [Table of Contents](#table-of-contents)
  - [About the Project](#about-the-project)
    - [Dataset description](#dataset-description)
    - [Processing](#processing)
    - [Feature extraction](#feature-extraction)
    - [Classification](#classification)
    - [Built with](#built-with)
    - [License](#license)
    - [Relevant Publications](#relevant-publications)
  - [Getting Started](#getting-started)
    - [Project structure](#project-structure)
  - [Contributing](#contributing)
  - [Contact](#contact)

<!-- ABOUT THE PROJECT -->
## About the Project

**Date**: June 2022

**Researchers**:

- Joeri Zwerts (j.a.zwerts@uu.nl)
- Heysem Kaya (h.kaya@uu.nl)

**Research Software Engineers**:

- Parisa Zahedi (p.zahedi@uu.nl)
- Casper Kaandorp (c.s.kaandorp@uu.nl)
- Jelle Treep (h.j.treep@uu.nl)

### Dataset description
The dataset for this project contains recordings in `.wav` format at 1 minute length and at a sample rate of 48000 samples/second. The recordings are taken at three locations:
- Chimpanze sanctuary
- Natural forest
- Semi-natural Chimanze enclosures  
The Chimpanze sanctuary and Natural forest are used for training and optimizing the classifiers. The Semi-natural Chimpanze recordings are used as an independent evaulation of the classifiers.

### Preprocessing 
The datasets are labeled into 2 classes (Chimpanze & background) using [Raven Pro](https://ravensoundsoftware.com/software/) annotation software, and extracted from the original recordings. Find scripts [here](./bioacoustics/1_wav_processing/raven_to_wav).

To speed up the labeling process we developed an energy-change based algorithm to filter out irrelevant parts of the recordings, see [Condensation](./bioacoustics/1_wav_processing/condensation).

To increase and diversify our training set we have created synthetic samples by embedding the sanctuary vocalizations into the recorded jungle audio that is labeled as 'background', see [Synthetic data](./bioacoustics/1_wav_processing/syntetic_data).

### Feature extraction
We trained the models on frames of 0.5 seconds.  
Before calculating features we apply a Butterworth bandpass filter with low cutoff at 100 Hz and a high cutoff at 2000 Hz.  
For classification using SVM we extract statistical features from different representations of the audio signal.  
For classification using Deep learning we use a mel spectrogram representation as input. 

| <img src="/img/melspectrogram.png" width="400" /> | 
|:--:| 
| *Chimpanze vocalization in mel spectrogram representation* |

### Classification
**SVM**  
From the 1140 statistical features from the previous step we select a normalized feature set of 50 features. The selection is based on feature importances computed with an Extra Trees Classifier. We train and optimize the SVM model on those 50 features using 'macro average recall' as evaluation criterion.
On the independent test set the SVM model establishes a 'macro average recall' of **0.84**.

**Deep learning**  
We trained several architectures of Convolutional Neural Networks (CNN) and a Residual network model (Resnet).

### Built with

- [Python 3.8](https://www.python.org/)
- [librosa](https://librosa.org/)
- [scikit-learn](https://scikit-learn.org/stable/index.html)
- [tensorflow](https://www.tensorflow.org/)

<!-- Do not forget to also include the license in a separate file(LICENSE[.txt/.md]) and link it properly. -->
### License

The code in this project is released under [Apache 2.0](LICENSE.md).

### Relevant publications

- Introducing a central african primate vocalisation dataset for automated species classification.\ 
Zwerts, J. A., Treep, J., Kaandorp, C. S., Meewis, F., Koot, A. C., & Kaya, H. (2021).\ 
[arXiv preprint](https://arxiv.org/pdf/2101.10390.pdf)
- The INTERSPEECH 2021 Computational Paralinguistics Challenge: COVID-19 cough, COVID-19 speech, escalation & primates.\
Schuller, B. W., Batliner, A., Bergler, C., Mascolo, C., Han, J., Lefter, I., ... & Kaandorp, C. (2021).\
[arXiv preprint](https://arxiv.org/pdf/2102.13468.pdf)


<!-- GETTING STARTED -->
## Getting Started
There are two situations in which you can directly apply the scripts in this repository and we tailored the documentation towards these situations:
1. You have audio data and a set of manual annotations (in e.g. txt or csv format) and want to use the whole pipeline (processing, augmentation, feature extraction and machine learning). 
2. You have a highly similar dataset and want to use one of our models to help find Chimpanze vocalizations.

If 1 applies to you, take a look at the project structure below and find getting started instructions for each step in the respective folders: [1_wav_processing](./bioacoustics/1_wav_processing), [2_feature_extraction](./bioacoustics/2_feature_extraction) and [3_classifier](./bioacoustics/3_classifier).

If 2 applies to you, go to step [3_classifier](./bioacoustics/3_classifier/README.md) and read the specific instructions for applying our models on your data.


### Project structure

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

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

To contribute:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<!-- CONTACT -->
## Contact

[Joeri Zwerts](https://www.uu.nl/medewerkers/JAZwerts) - j.a.zwerts@uu.nl

[Research Engineering team](https://utrechtuniversity.github.io/research-engineering/) - research.engineering@uu.nl

Project Link: [https://github.com/UtrechtUniversity/animal-sounds](https://github.com/UtrechtUniversity/animal-sounds)
