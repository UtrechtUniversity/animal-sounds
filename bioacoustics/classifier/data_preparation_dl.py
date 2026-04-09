"""Script to prepare train and test data to feed into a model"""

import glob
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


def read_features(features_dir):
    """Reads all feature files and consolidate them into a dataframe
    Parameters
    ----------
    features_dir: str
        The path of all feature files.

    Returns
    -------
    DataFrame:
        a dataframe of all features
    """
    features_path = os.path.join(features_dir, "**", "*.pkl")
    files = glob.glob(features_path, recursive=True)

    dfs = []
    for file_path in files:
        df = pd.read_pickle(file_path)
        dfs.append(df)

    df_result = pd.concat(dfs, join="inner").reset_index(drop=True)
    logging.info("df_result.shape: %s", df_result.shape)
    return df_result


def get_dl_format(df, le=None, without_label=False):
    specs = []
    for feat in df["features"]:
        if isinstance(feat, list):
            if len(feat) == 1:
                spec = feat[0]
            else:
                spec = np.stack(feat, axis=0)
        else:
            spec = np.array(feat) if not isinstance(feat, np.ndarray) else feat

        specs.append(spec)

    if without_label:
        return specs, None, le

    if le is None:
        le = LabelEncoder()
        labels = le.fit_transform(df["label_1"])
    else:
        labels = le.transform(df["label_1"])

    return specs, labels, le


def prepare_data_dl(features_dir, mode="train", val_size=0.15, test_size=0.15, seed=42):
    """
    Prepare datasets for training, evaluation, or prediction.

    Parameters
    ----------
    features_dir : str
        Path to feature files (.pkl).
    mode : str
        'train', 'evaluate', or 'predict'.
    val_size : float
        Fraction for validation (train mode only).
    test_size : float
        Fraction for test (train mode only).
    seed : int
        Random seed for reproducible splits.

    Returns
    -------
    x_train, y_train, x_val, y_val, x_test, y_test
    """
    df = read_features(features_dir)

    if mode == "predict":
        specs, _, le = get_dl_format(df, without_label=True)
        return None, None, None, None, specs, None

    if mode == "evaluate":
        specs, labels, le = get_dl_format(df, without_label=False)
        return None, None, None, None, specs, labels

    if mode == "train":
        le = LabelEncoder()
        labels = le.fit_transform(df["label_1"])
        logging.info("classes: %s", list(le.classes_))

        train_idx, temp_idx = train_test_split(
            np.arange(len(df)),
            test_size=val_size + test_size,
            stratify=labels,
            random_state=seed,
        )

        val_ratio = val_size / (val_size + test_size)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=1 - val_ratio,
            stratify=labels[temp_idx],
            random_state=seed,
        )

        x_train, y_train, _ = get_dl_format(df.iloc[train_idx], le=le)
        x_val, y_val, _ = get_dl_format(df.iloc[val_idx], le=le)
        x_test, y_test, _ = get_dl_format(df.iloc[test_idx], le=le)

        return x_train, y_train, x_val, y_val, x_test, y_test
