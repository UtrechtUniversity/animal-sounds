# 3_classifier

## Table of Contents

- [How does it work](#how-does-it-work)
- [Software dependencies](#software-requirements)
- [Usage](#basic-usage)
    - [Train](#train)
    - [Predict](#predict)
    - [Shell script](#shell-script)
- [Remarks](#remarks)

## How does it work

### Preprocessing

### Feature selection (SVM)
With the total of 1140 acoustic features from the training samples as input, we used a Recursive Feature Elimination (RFE) technique to determine an optimal number of features. This technique yielded a broad optimum with centered a round a rough estimate of 50 features.
| <img src="/results/RFE.png" width="400" /> | 
|:--:| 
| *Recursive Feature Elimination on training dataset* |

In the preprocessing steps of the SVM model, we perform feature selection using [Extra Trees Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html) to rank the features and select the 50 most important features.


### Training


### Predicting



## Software requirements

- Python 3
- [Sklearn ~1.1.1](https://scikit-learn.org/)
- [Numpy ~1.20.3](https://numpy.org/)
- [Pandas ~1.3.4](https://pandas.pydata.org)
- [Tensorflow ~2](https://www.tensorflow.org)
- [Keras](https://keras.io/)


## Basic usage


### Train

```
python3.8 train.py --model=svm
            --feature_dir=../../output/features/
            --output_dir=../../output/models/svm/
```

### Predict

```
python3.8 predict.py --model=svm
            --feature_dir=../../output/features/
            --trained_model_path=../../output/models/svm/svm_model.sav
            --output_dir=../../output/models/svm/predictions/
```


### Shell script


## Remarks



