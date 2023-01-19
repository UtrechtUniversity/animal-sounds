"""Script to prepare train and test data to feed into a model"""
import glob
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


def read_features(features_path):
    """Reads all feature files and consolidate them into a dataframe
    Parameters
    ----------
    features_path: str
        The path of all feature files.

    Returns
    -------
    DataFrame:
        a dataframe of all features
    """
    files = glob.glob(features_path)
    df_result = pd.DataFrame(columns=["file_path", "features", "label_1"])
    for file_path in files:
        df_features = pd.read_pickle(file_path).copy()
        df_result = pd.concat([df_result, df_features], join="inner").copy()

    print("df_result.shape:", df_result.shape)

    return df_result


def get_dl_format(df_features, labels):
    """Convert features and corresponding classification labels into numpy
    arrays and categorical values

    Parameters
    ----------
    df_features: DataFrame
        Dataframe of features
    labels: Series
        Include labels
    Returns
    -------
    DataFrame:
        a dataframe of x , yy
    """

    x_values = np.array(df_features.features.tolist())

    yy_values = None
    if labels is not None:
        y_values = np.array(labels.tolist())

        # Encode the classification labels
        le_values = LabelEncoder()
        yy_values = to_categorical(le_values.fit_transform(y_values))
        print("yy.shape", yy_values.shape)
        print("category classes:)", list(le_values.classes_))

    return x_values, yy_values


def normalize_data(x_values, x_mean=None, x_std=None, normval_fp="", train_mode=True):
    """Normalize the given dataset

    Parameters
    ----------
    x: DataFrame
        Dataframe of features
    x_mean: float
        Includes mean value of training dataset to apply on test dataset. None
         for training dataset.
    x_std: float
        Includes std value of training dataset to apply on test dataset. None
         for training dataset.
    normval_fp: str
        File path to save normalizing values or reading from
    train_mode: bool
        Indicates if it is in training mode or not
    """
    x_mean = x_values.mean() if x_mean is None else x_mean
    x_std = x_values.std() if x_std is None else x_std
    if train_mode:
        norm_df = pd.DataFrame({"x_mean": [x_mean], "x_std": [x_std]})
        norm_df.to_csv(normval_fp, index=False)

    x_values = (x_values - x_mean) / x_std

    return x_values


def get_dataset(features_path, without_label, fraction=1):
    """Read features and set it in a dl format

    Parameters
    ----------
    features_path: str
        The path of all feature files.
    without_label: bool
        Indicates if labels are available in the dataset.
        E.g. In test set no label could be available
    fraction: float
        The percentage of given data that should be returned.
        E.g. for hyper parameter optimization we use only 60% of the given
        dataset
    Returns
    -------
    DataFrames:
        x_frac, y_frac
    """
    df_features = read_features(features_path)
    x_values = df_features[df_features.columns.difference(["file_path", "label_1"])]
    y_values = None if without_label else df_features["label_1"]
    x_values, y_values = get_dl_format(x_values, y_values)

    x_frac = x_values[0 : int(fraction * len(x_values))]
    y_frac = y_values[0 : int(fraction * len(y_values))]

    return x_frac, y_frac


def prepare_data_dl(
    features_path,
    without_label=False,
    trained_model_path="",
    norm_val_dir="",
    fraction=1,
):
    """Prepare train and test dataset

    Parameters
    ----------
    features_path: str
        The path of all feature files.
    without_label: bool
        Indicates if labels are available in the dataset.
        E.g. In test set no label could be available
    trained_model_path: str
        The file path of a trained model.
        E.g. for prediction purposes
    norm_val_dir: str
        The file path of normalization values, created based on training and
        used for test dataset.
    fraction: float
        The percentage of given data that should be returned.
        E.g. for hyper parameter optimization we use only 60% of the given
        dataset
    Returns
    -------
    Dataframes:
        X_train, y_train, X_test, y_test in four different dataframe
    """

    x_values, y_values = get_dataset(features_path, without_label, fraction)
    normval_fp = os.path.join(norm_val_dir, "normval.csv")

    test_size = 0.2

    if trained_model_path == "":  # prepare data for train model scenario
        # split dataset to train and test set

        x_train = x_values[0 : int(round((1 - test_size) * len(x_values)))]
        y_train = y_values[0 : int(round((1 - test_size) * len(y_values)))]
        x_test = x_values[int(round((1 - test_size) * len(x_values))) :]
        y_test = y_values[int(round((1 - test_size) * len(x_values))) :]

        x_train = normalize_data(x_train, normval_fp=normval_fp, train_mode=True)
        normdf = pd.read_csv(normval_fp)
        x_test = normalize_data(
            x_test,
            x_mean=normdf.loc[0, "x_mean"],
            x_std=normdf.loc[0, "x_std"],
            train_mode=False,
        )
    else:  # prepare data to apply an existing model

        x_test = x_values.copy()
        normdf = pd.read_csv(normval_fp)
        x_test = normalize_data(
            x_test,
            x_mean=normdf.loc[0, "x_mean"],
            x_std=normdf.loc[0, "x_std"],
            train_mode=False,
        )

        x_train = None
        y_train = None

        # apply model on un-labeled dataset
        y_test = None if without_label else y_values

    return x_train, y_train, x_test, y_test
