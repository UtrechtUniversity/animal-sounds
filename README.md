# animal-sounds

<!-- Include Github badges here (optional) -->
<!-- e.g. Github Actions workflow status -->

Description of this (technical) project and of the project itself.

<!-- TABLE OF CONTENTS -->
## Table of Contents

- [Project Title](#project-title)
  - [Table of Contents](#table-of-contents)
  - [About the Project](#about-the-project)
    - [Built with](#built-with)
    - [License](#license)
    - [Attribution and academic use](#attribution-and-academic-use)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Usage](#usage)
    - [Subsection](#subsection)
  - [Links](#links)
  - [Contributing](#contributing)
  - [Notes](#notes)
  - [Contact](#contact)

<!-- ABOUT THE PROJECT -->
## About the Project

**Date**: October 2021

**Researcher(s)**:

- Joeri Zwerts (researcher.1@uu.nl)
- Heysem Kaya (researcher.2@uu.nl)

**Research Software Engineer(s)**:

- Parisa Zahedi (p.zahedi@uu.nl)
- Casper Kaandorp (c.s.kaandorp@uu.nl)
- Jelle Treep (h.j.treep@uu.nl)

### Built with

- [Python 3.8](https://www.python.org/)
- [librosa](https://librosa.org/)
- [scikit-learn](https://scikit-learn.org/stable/index.html)

<!-- Do not forget to also include the license in a separate file(LICENSE[.txt/.md]) and link it properly. -->
### License

The code in this project is released under [Apache 2.0](LICENSE.md).

### Attribution and academic use

Relevant publications:

- Introducing a central african primate vocalisation dataset for automated species classification.\ 
Zwerts, J. A., Treep, J., Kaandorp, C. S., Meewis, F., Koot, A. C., & Kaya, H. (2021).\ 
[arXiv preprint](https://arxiv.org/pdf/2101.10390.pdf)
- The INTERSPEECH 2021 Computational Paralinguistics Challenge: COVID-19 cough, COVID-19 speech, escalation & primates.\
Schuller, B. W., Batliner, A., Bergler, C., Mascolo, C., Han, J., Lefter, I., ... & Kaandorp, C. (2021).\
[arXiv preprint](https://arxiv.org/pdf/2102.13468.pdf)


<!-- GETTING STARTED -->
## Getting Started



### Prerequisites

To install and run this project you need to have the following prerequisites installed.

- Python

### Installation

To run the project, ensure to install the project's dependencies.

```sh
pip install -r requirements.txt
```

<!-- USAGE -->
## Usage
7 steps are used to come from raw data to classification results:
### 0. Manual annotation on raw audio data
Generate a representative set of annotations of the species of interest to tune parameters for extracting sections of interest from the larger body of audio files. [Raven](https://ravensoundsoftware.com/) is used in this project for manual annotations. Generate a background dataset as well by manually annotation (or randomly sample sections in between species annotations).

### 1. Extraction of audio sections of interest
Sections of interest are identified based on increased energy in certain frequencies. This results in a removal of ~85% of irrelevant sections of plain jungle background (and lowers the annotation effort). <1% of annotations of step 1 are lost in this step.
[Instructions](1_condensation/README.md)

### 1b. Manual annotation on processed audiodata
Continue manual annotations on the 'condensed' WAV files until a substantial set of annotations is reached (e.g 500 vocalizations) for machine learning.

### 2. Processing of raw data to annotations
Create WAV files of the species and background annotations only. [Instructions](2_wav_processing/README.md)

### 3. Data augmentation
Generate synthetic data by adding background signal to annotations in varying proportions. [Instructions](3_synthetic_data/README.md)

### 4. Feature extraction
Extract features (e.g. MFCC, RASTA-PLPC) from annotations and store in .csv format.
[Further details and instructions](4_feature_extraction/README.md)

### 5. Modelling
Separate train, validation and test sets and optimize performance of CNN and SVM models. The SVM model uses a subset of the most important features from step 4 for training and testing. CNN is trained on audio data from step 3 and uses CNN features.
[Further details and instructions](5_classifier/README.md)

### 6. Prediction
[Instructions](6_prediction/README.md)


<!-- LINKS -->
## Links


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

To contribute:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- NOTES -->
## Notes

Additional notes on the project can be left here.

<!-- CONTACT -->
## Contact

Contact Name - [@twitterhandle](https://twitter.com/username) - contact.rse@uu.nl

Project Link: [https://github.com/UtrechtUniversity/animal-sounds](https://github.com/UtrechtUniversity/animal-sounds)
