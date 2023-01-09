"""Script to prepare train and test data to feed into a model"""
import pandas as pd
import glob
import numpy as np
import os

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
    for fp in files:
        df_features = pd.read_pickle(fp).copy()
        df_result = pd.concat([df_result, df_features], join="inner").copy()

    print("df_result.shape:", df_result.shape)

    return df_result


def get_dl_format(df_features, labels, num_channels=3):
    """Convert features and corresponding classification labels into
    numpy arrays and categorical values

    Parameters
    ----------
    df_features: DataFrame
        Dataframe of features
    labels: Series
        Include labels
    num_channels: int
         Number of channels
    Returns
    -------
    DataFrame:
        a dataframe of x , yy
    """

    x = np.array(df_features.features.tolist())
    # if num_channels == 1:
    #     x = x[..., np.newaxis]

    yy = None
    if labels is not None:
        y = np.array(labels.tolist())

        # Encode the classification labels
        le = LabelEncoder()
        yy = to_categorical(le.fit_transform(y))
        print("yy.shape", yy.shape)
        print("category classes:)", list(le.classes_))

    return x, yy


def normalize_data(x, x_mean=None, x_std=None, normval_fp="", train_mode=True):
    """
    mean and std is calculated
    :param x:
    :param x_mean:
    :param x_std:
    :param normval_fp:
    :return:
    """
    x_mean = x.mean() if x_mean is None else x_mean  # statistics.mean(x)
    x_std = x.std() if x_std is None else x_std  # std(x)
    if train_mode:
        norm_df = pd.DataFrame({"x_mean": [x_mean], "x_std": [x_std]})
        norm_df.to_csv(normval_fp, index=False)

    x = (x - x_mean) / x_std

    return x


def get_dataset(features_path, without_label, num_channels, fraction=1):
    df_features = read_features(features_path)
    x = df_features[df_features.columns.difference(["file_path", "label_1"])]
    y = None if without_label else df_features["label_1"]
    x, y = get_dl_format(x, y, num_channels)

    x_frac = x[0 : int(fraction * len(x))]
    y_frac = y[0 : int(fraction * len(y))]

    return x_frac, y_frac


def prepare_data_dl(
    features_path,
    without_label=False,
    trained_model_path="",
    num_channels=3,
    normval_dir="",
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
    num_channels: int
         Number of channels
    Returns
    -------
    DataFrames:
         X_train, y_train, X_test, y_test in four different dataframe
    """
    # NOTE: REMOVE THIS PART?
    # df_features = read_features(features_path)
    # x = df_features[df_features.columns.difference(['file_path', 'label_1'])]
    # y = None if without_label else df_features['label_1']
    # x, y = get_dl_format(x, y, num_channels)

    x, y = get_dataset(features_path, without_label, num_channels, fraction)
    normval_fp = os.path.join(normval_dir, "normval.csv")

    # NOTE: REMOVE THIS PART?
    # if trained_model_path == '':  # prepare data for train model scenario
    #     # split dataset to train and test set
    #     X_train, X_test, y_train, y_test = model_selection.train_test_split(x,
    #                                                                         y,
    #                                                                         test_size=0.2,
    #                                                                         random_state=1)
    #
    #     X_train = normalize_data(X_train, normval_fp=normval_fp, train_mode=True)
    #     normdf = pd.read_csv(normval_fp)
    #     X_test = normalize_data(X_test,
    #                             x_mean=normdf.loc[0,'x_mean'],
    #                             x_std=normdf.loc[0,'x_std'],
    #                             train_mode=False)
    # else:  # prepare data to apply an existing model
    #
    #     X_test = x.copy()
    #     normdf = pd.read_csv(normval_fp)
    #     X_test = normalize_data(X_test,
    #                             x_mean=normdf.loc[0, 'x_mean'],
    #                             x_std=normdf.loc[0, 'x_std'],
    #                             train_mode=False)
    #
    #     X_train = None
    #     y_train = None
    #
    #     y_test = None if without_label else y  # apply model on un-labeled dataset

    test_size = 0.2

    if trained_model_path == "":  # prepare data for train model scenario
        # split dataset to train and test set

        X_train = x[0 : int(round((1 - test_size) * len(x)))]
        y_train = y[0 : int(round((1 - test_size) * len(y)))]
        X_test = x[int(round((1 - test_size) * len(x))) :]
        y_test = y[int(round((1 - test_size) * len(x))) :]

        X_train = normalize_data(X_train, normval_fp=normval_fp, train_mode=True)
        normdf = pd.read_csv(normval_fp)
        X_test = normalize_data(
            X_test,
            x_mean=normdf.loc[0, "x_mean"],
            x_std=normdf.loc[0, "x_std"],
            train_mode=False,
        )
    else:  # prepare data to apply an existing model

        X_test = x.copy()
        normdf = pd.read_csv(normval_fp)
        X_test = normalize_data(
            X_test,
            x_mean=normdf.loc[0, "x_mean"],
            x_std=normdf.loc[0, "x_std"],
            train_mode=False,
        )

        X_train = None
        y_train = None

        y_test = None if without_label else y  # apply model on un-labeled dataset

    return X_train, y_train, X_test, y_test
