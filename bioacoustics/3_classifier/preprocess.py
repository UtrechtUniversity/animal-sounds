import pandas as pd
import numpy as np
import glob

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

from sklearn.ensemble import ExtraTreesClassifier
from matplotlib import pyplot

from model.svm_model import SVM_model

def read_features(features_path, index):
    files = glob.glob(features_path)
    return pd.read_csv(files[index])


def normalize(x_train, x_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled


def feature_importance(x_train, y_train):
    model = ExtraTreesClassifier(n_estimators=10)
    return model.fit(x_train, y_train)


def feature_selection(x_train, x_test, columnnames, model, output_dir, numfeat):
    df = pd.DataFrame({'featname': columnnames, 'feature_importances': model.feature_importances_})
    df.to_csv(output_dir + 'feature_importances.csv')
    indices = list(df.sort_values(by=['feature_importances'], ascending=False).index[0:numfeat])
    return x_train[:, indices], x_test[:, indices]


def recursive_features(X, y, columnnames, output_dir):
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    # Create the RFE object and compute a cross-validated score.
    svc = SVC(kernel="linear")
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
                  scoring='recall_macro', n_jobs=-1)
    rfecv.fit(X, y)

    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    pyplot.figure()
    pyplot.xlabel("Number of features selected")
    pyplot.ylabel("Cross validation score (nb of correct classifications)")
    pyplot.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    pyplot.show()

    df = pd.DataFrame({'featname': columnnames, 'feature_ranking': rfecv.ranking_})
    df.to_csv(output_dir + 'feature_rankings.csv')
    print(rfecv.ranking_)


def run_svm(x_train, y_train, x_test, y_test, output_dir):
    s = SVM_model(x_train, y_train, x_test, y_test)
    s.run_model()
    s.save_results(output_dir)


def split_test(features_path, index, dim, test_size=0.2):
    subset = 1
    df = read_features(features_path, index)
    x_train = df.iloc[0:int(round((1 - test_size) * len(df))), dim[0]:dim[1]]
    y_train = df['label_1'][0:int(round((1 - test_size) * len(df)))]
    x_test = df.iloc[int(round((1 - test_size) * len(df))):-1, dim[0]:dim[1]]
    y_test = df['label_1'][int(round((1 - test_size) * len(df))):-1]
    y_file = df['file_path'][int(round((1 - test_size) * len(df))):-1]

    if subset and x_train.shape[0] > 10000:
        sample_idx = np.random.choice(x_train.shape[0], replace=False, size=10000)
        x_train = x_train.iloc[sample_idx]
        y_train = y_train.iloc[sample_idx]

    return x_train, x_test, y_train, y_test, y_file


def main(ml_method):
    features_path = '/home/jelle/Repositories/animalsounds/data/features/features_v7/features_sanctsynth/*'
    output_dir = '/home/jelle/Repositories/animalsounds/data/svm_results/training_alldata/'
    feat = 'all'


    if feat == 'general':
        dim = [5, 101]
    elif feat == 'mfcc':
        dim = [101, 491]
    elif feat == 'mfcc+rasta':
        dim = [101, -1]
    else:
        dim = [5, -1]

    for i in range(len(glob.glob(features_path))):
        print(i)
        if i == 0:
            x_train, x_test, y_train, y_test, y_file = split_test(features_path, i, dim)
        else:
            temp_x_train, temp_x_test, temp_y_train, temp_y_test, temp_y_file = split_test(features_path, i, dim)
            x_train = pd.concat([x_train, temp_x_train], sort=False)
            x_test = pd.concat([x_test, temp_x_test], sort=False)
            y_train = pd.concat([y_train, temp_y_train], sort=False)
            y_test = pd.concat([y_test, temp_y_test], sort=False)
            y_file = pd.concat([y_file, temp_y_file], sort=False)


    x_train, x_test = normalize(x_train.to_numpy(), x_test.to_numpy())
    y_file.to_csv(output_dir + 'test_files.csv')


    model = feature_importance(x_train, y_train)
    x_train, x_test = feature_selection(x_train, x_test, temp_x_test.columns, model, output_dir, numfeat=50)
    if ml_method == 'svm':
        run_svm(x_train, y_train, x_test, y_test, output_dir)
    elif ml_method == 'rec_feat':
        recursive_features(x_train, y_train, temp_x_test.columns, output_dir)


if __name__ == "__main__":
    method = 'svm'  # 'rec_feat' or 'svm'
    main(method)
