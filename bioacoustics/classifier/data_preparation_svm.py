"""Data preparation for SVM classifier."""
import pandas as pd
import numpy as np
import glob
import os

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import ExtraTreesClassifier
from matplotlib import pyplot
from pickle import dump, load


def read_features(features_path, index):
    """Read feature files.

    This function reads feature files (.csv) and returns
    Pandas DataFrame containing the features.

    Parameters
    ----------
    features_path : str
        Specifies the location of the files that contain the features.

    index : int
        index defining the file in the list of files

    Returns
    -------
    DataFrame:
        a dataframe of all features
    """
    files = glob.glob(features_path + "**/*.csv", recursive=True)
    return pd.read_csv(files[index])


def normalize_fit(x_train, output_dir, x_test=None):
    """Fit normalizer model and transform values.

    This function fits a scaler model to normalize all feature values
    and transforms the values. If x_test is specified, the model is stored
    to be reused for prediction purposes.

    Parameters
    ----------
    x_train : DataFrame
        dataframe containing original feature values

    output_dir: str
        Specifies the location where the scaler model is saved.

    x_test : DataFrame
        dataframe containing original feature values of test set.

    Returns
    -------
    DataFrame:
        a dataframe containing normalized feature values
    """
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    if x_test is not None:
        dump(scaler, open(output_dir + "scaler/scaler.pkl", "wb"))
        x_test_scaled = scaler.transform(x_test)
        return x_train_scaled, x_test_scaled
    else:
        return x_train_scaled


def normalize(x_test, svm_dir):
    """Normalize feature values for prediction.

    This function loads a previously saved scaler model
    and uses it to normalize all feature values.

    Parameters
    ----------
    x_test : DataFrame
        dataframe containing original feature values

    svm_dir: str
        Specifies the location where the scaler model is saved.

    Returns
    -------
    DataFrame:
        a dataframe containing normalized feature values
    """
    scaler = load(open(svm_dir + "scaler/scaler.pkl", "rb"))
    x_test_scaled = scaler.transform(x_test)
    return x_test_scaled


def read_file(file, dim):
    """Read features and labels.

    This function reads features and class labels from feature file (.csv).

    Parameters
    ----------
    file : str
        File path of the features file.

    dim: list
        First and last column containing features

    Returns
    -------
    DataFrames:
        DataFrames containing features (x) and labels (y)
    """
    df = pd.read_csv(file)
    x = df.iloc[:, dim[0] : dim[1]]
    y = df["label_1"]
    return x, y


def filter_features(data, file, numfeat=50):
    """Select best features.

    This function reads a .csv file with feature importances
    (see feature selection function) and selects the 50
    (or otherwise specified with numfeat) most 'important' features
    and returns them in a dataframe.

    Parameters
    ----------
    data : DataFrame
        Original DataFrame containing all features

    file : str
        File path of the feature importances table.

    numfeat: int
        number of features to select

    Returns
    -------
    DataFrame:
        DataFrames containing selected features
    """

    df = pd.read_csv(file)
    indices = list(
        df.sort_values(by=["feature_importances"], ascending=False).index[0:numfeat]
    )
    return data[:, indices]


def read_files(file_path, dim, svm_dir, predict=False, hoplength=0.25, output_dir=""):
    """Read and filter feature data.

    This function reads features and labels from all relevant files
    containing features, filters the 'most important' features and
    combines them in one DataFrame.

    Parameters
    ----------
    file_path : str
        File path of the features file.

    dim: list
        First and last column containing features

    svm_dir: str
        Directory path of the feature importances file

    predict: bool, default False
        True when calling the function from predict.py

    hoplength: float
        Used when calling the function from predict.py
        to determine the offset of each frame
    output_dir: str, optional
        Used when calling the function from predict.py
        to specify the location of the output file

    Returns
    -------
    DataFrames:
        DataFrames containing features (x) and labels (y)
    """
    feature_file = svm_dir + "feature_importances.csv"
    files = glob.glob(file_path + "**/*.csv", recursive=True)
    if predict is True:
        dim = [0, -1]
    for i in range(len(files)):
        if i == 0:
            x, y = read_file(files[i], dim)
        else:
            temp_x, temp_y = read_file(files[i], dim)
            x = pd.concat([x, temp_x], sort=False)
            y = pd.concat([y, temp_y], sort=False)

    if predict is True:
        # create info file to match predictions with actual audio
        frame_info = x.iloc[:, 0:5]
        frame_info["offset[s]"] = frame_info["frameId"] * hoplength
        print(list(frame_info.columns))
        frame_info.drop(
            columns=["label_1", "label_2", "frameId", "length[s]"], inplace=True
        )
        print(frame_info)
        frame_info.to_csv(output_dir + "frame-info.csv")
        x = x.iloc[:, 5:-1]

    x = filter_features(x.to_numpy(), feature_file)

    return x, y


def feature_importance(x_train, y_train):
    """Create feature importance model.

    This function create a feature importance model using the
    ExtraTreesClassifier method.

    Parameters
    ----------
    x_train: DataFrame
        Features of training set

    y_train: DataFrame
        Class labels of training set

    Returns
    -------
    model:
        ExtraTreesClassifier model
    """
    model = ExtraTreesClassifier(n_estimators=10)
    return model.fit(x_train, y_train)


def feature_selection(x_train, x_test, columnnames, model, output_dir, numfeat):
    """Save feature importance scores and filter feature sets.

    This function saves feature importance scores in a .csv files.
    It then filters and returns the most important features from
    the original training and test feature sets.

    Parameters
    ----------
    x_train: DataFrame
        Features of training set

    x_test: DataFrame
        Features of test set

    columnnames: list
        Feature names

    model: object
        Fitted estimator

    output_dir: str
        Directory path where the feature importances file will be stored

    numfeat: int
        Number of features to select

    Returns
    -------
    DataFrames:
        DataFrames containing filtered features of training set (x_train) and test set (x_test)
    """
    df = pd.DataFrame(
        {"featname": columnnames, "feature_importances": model.feature_importances_}
    )
    df.to_csv(output_dir + "feature_importances.csv")
    indices = list(
        df.sort_values(by=["feature_importances"], ascending=False).index[0:numfeat]
    )
    return x_train[:, indices], x_test[:, indices]


def recursive_features(X, y, columnnames, output_dir):
    """Recursive feature elimination.

    This function runs recursive feature elimination and plots cross
    validation scores as a function of number of features.

    This function is only used for explorative purposes and as a basis to
    set the number of features to select in the ExtraTreesClassifier method.

    Parameters
    ----------
    X: DataFrame
        Features of training set

    y: DataFrame
        Class labels of training set

    columnnames: list
        Feature names

    output_dir: str
        Directory path where the feature rankings will be stored
    """
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    # Create the RFE object and compute a cross-validated score.
    svc = SVC(kernel="linear")
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    rfecv = RFECV(
        estimator=svc, step=1, cv=StratifiedKFold(2), scoring="recall_macro", n_jobs=-1
    )
    rfecv.fit(X, y)

    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    pyplot.figure()
    pyplot.xlabel("Number of features selected")
    pyplot.ylabel("Cross validation score")
    pyplot.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    pyplot.show()

    df = pd.DataFrame({"featname": columnnames, "feature_ranking": rfecv.ranking_})
    df.to_csv(output_dir + "feature_rankings.csv")
    print(rfecv.ranking_)


def split_test(features_path, index, dim, test_size=0.2):
    """Split data in train and test datasets.

    This function splits the datasets used for the training phase in a
    training and test dataset. The test set always consists of the bottom
    x rows of the dataframe. The size of the test dataset can be
    altered by changing the test_size fraction. The maximum size of the
    train set is limited to 10k samples taken randomly from the training set.

    Parameters
    ----------
    features_path: str
        File path of the features files.

    index: int
        index defining the file in the list of files

    dim: list
        First and last column containing features

    test_size: float
        Fraction of data to be used for the test set.

    Returns
    -------
    DataFrames:
        Dataframes containing features of the train (x_train) and test (x_test)
        sets, class labels (y_train, y_test) and filenames (y_file)
    """
    subset = 1
    df = read_features(features_path, index)
    x_train = df.iloc[0 : int(round((1 - test_size) * len(df))), dim[0] : dim[1]]
    y_train = df["label_1"][0 : int(round((1 - test_size) * len(df)))]
    x_test = df.iloc[int(round((1 - test_size) * len(df))) : -1, dim[0] : dim[1]]
    y_test = df["label_1"][int(round((1 - test_size) * len(df))) : -1]
    y_file = df["file_path"][int(round((1 - test_size) * len(df))) : -1]

    if subset and x_train.shape[0] > 10000:
        sample_idx = np.random.choice(x_train.shape[0], replace=False, size=10000)
        x_train = x_train.iloc[sample_idx]
        y_train = y_train.iloc[sample_idx]

    return x_train, x_test, y_train, y_test, y_file


def prepare_data_svm(features_path, output_dir, trained_model_path=""):
    """Preprocess data for SVM training and prediction.

    This main function prepares training and testing features and class
    labels for training and predictions using Support Vector Machines.

    Parameters
    ----------
    features_path: str
        File path of the features files.

    output_dir: str
        Directory path where the feature rankings will be stored

    trained_model_path: str
        Directory path where the trained model will be store (or loaded from)

    Returns
    -------
    DataFrames:
        Dataframes containing features of the train (x_train) and test (x_test)
        sets and class labels (y_train, y_test)
    """

    print(features_path + "**/*.csv")
    print(glob.glob(features_path + "**/*.csv", recursive=True))

    feat = "all"
    if feat == "general":
        dim = [5, 101]
    elif feat == "mfcc":
        dim = [101, 491]
    elif feat == "mfcc+rasta":
        dim = [101, -1]
    else:
        dim = [5, -1]
    if trained_model_path == "":
        # create training and test sets
        for i in range(len(glob.glob(features_path + "**/*.csv", recursive=True))):
            print(i)
            if i == 0:
                x_train, x_test, y_train, y_test, y_file = split_test(
                    features_path, i, dim
                )
            else:
                (
                    temp_x_train,
                    temp_x_test,
                    temp_y_train,
                    temp_y_test,
                    temp_y_file,
                ) = split_test(features_path, i, dim)
                x_train = pd.concat([x_train, temp_x_train], sort=False)
                x_test = pd.concat([x_test, temp_x_test], sort=False)
                y_train = pd.concat([y_train, temp_y_train], sort=False)
                y_test = pd.concat([y_test, temp_y_test], sort=False)
                y_file = pd.concat([y_file, temp_y_file], sort=False)

        y_file.to_csv(output_dir + "test_files.csv")
        # recursive_features(x_train, y_train, temp_x_test.columns, output_dir)

        # normalize first time for feature selection purposes
        x_train_tmp = normalize_fit(x_train.to_numpy(), output_dir)
        model = feature_importance(x_train_tmp, y_train)
        x_train, x_test = feature_selection(
            x_train.to_numpy(),
            x_test.to_numpy(),
            temp_x_test.columns,
            model,
            output_dir,
            numfeat=50,
        )

        # normalize the 50 features of interest of x_train and x_test
        x_train, x_test = normalize_fit(x_train, output_dir, x_test)
    else:
        # create only test set
        trained_model_dir = os.path.split(trained_model_path)[0] + "/"
        x_test, y_test = read_files(
            features_path,
            [0, -1],
            trained_model_dir,
            predict=True,
            output_dir=output_dir,
        )
        x_test = normalize(x_test, trained_model_dir)
        x_train = None
        y_train = None

    return x_train, y_train, x_test, y_test
