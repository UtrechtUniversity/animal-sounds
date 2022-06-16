import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from model.svm_model import SVM_model

import sklearn.model_selection as model_selection
from sklearn.ensemble import ExtraTreesClassifier
from pickle import dump
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from matplotlib import pyplot


def read_features(features_path, index):
    files = glob.glob(features_path + '**/*.csv', recursive=True)
    return pd.read_csv(files[index])


def normalize(x_train, output_dir, x_test=None):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    if x_test is not None:
        dump(scaler, open(output_dir + 'scaler/scaler.pkl', 'wb'))
        x_test_scaled = scaler.transform(x_test)
        return x_train_scaled, x_test_scaled
    else:
        return x_train_scaled


def feature_importance(x_train, y_train):
    model = ExtraTreesClassifier(n_estimators=10)
    return model.fit(x_train, y_train)


def feature_selection(x_train, x_test, columnnames, model, output_dir, numfeat):
    df = pd.DataFrame({'featname': columnnames, 'feature_importances': model.feature_importances_})
    df.to_csv(output_dir + 'feature_importances.csv')
    indices = list(df.sort_values(by=['feature_importances'], ascending=False).index[0:numfeat])
    return x_train[:, indices], x_test[:, indices]


def get_models():
    models = dict()
    # lr
    rfe = RFE(estimator=LogisticRegression(), n_features_to_select=5)
    model = DecisionTreeClassifier()
    models['lr'] = Pipeline(steps=[('s', rfe), ('m', model)])
    # perceptron
    rfe = RFE(estimator=Perceptron(), n_features_to_select=5)
    model = DecisionTreeClassifier()
    models['per'] = Pipeline(steps=[('s', rfe), ('m', model)])
    # cart
    rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=5)
    model = DecisionTreeClassifier()
    models['cart'] = Pipeline(steps=[('s', rfe), ('m', model)])
    # rf
    rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=5)
    model = DecisionTreeClassifier()
    models['rf'] = Pipeline(steps=[('s', rfe), ('m', model)])
    # gbm
    rfe = RFE(estimator=GradientBoostingClassifier(), n_features_to_select=5)
    model = DecisionTreeClassifier()
    models['gbm'] = Pipeline(steps=[('s', rfe), ('m', model)])
    return models


def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='recall_macro', cv=cv, n_jobs=-1)
    return scores


def recursive_features_modeltest(x_train, y_train):
    # Build a classification task using 3 informative features
    le = LabelEncoder()
    le.fit(y_train)
    y = le.transform(y_train)
    # get the models to evaluate
    models = get_models()
    # evaluate the models and store results
    results, names = list(), list()
    for name, model in models.items():
        scores = evaluate_model(model, x_train, y)
        results.append(scores)
        names.append(name)
        print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
        # plot model performance for comparison
        pyplot.boxplot(results, labels=names, showmeans=True)
        pyplot.show()


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

# def run_svm(x_train, y_train, x_test, y_test, output_dir):
#     s = SVM_model(x_train, y_train, x_test, y_test)
#     s.run_model()
#     s.save_results(output_dir)


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


def prepare_data_svm(features_path, output_dir):
    print(features_path + '**/*.csv')
    print(glob.glob(features_path + '**/*.csv', recursive=True))

    feat = 'all'
    if feat == 'general':
        dim = [5, 101]
    elif feat == 'mfcc':
        dim = [101, 491]
    elif feat == 'mfcc+rasta':
        dim = [101, -1]
    else:
        dim = [5, -1]

    for i in range(len(glob.glob(features_path + '**/*.csv', recursive=True))):
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

    y_file.to_csv(output_dir + 'test_files.csv')
    # recursive_features(x_train, y_train, temp_x_test.columns, output_dir)

    # normalize first time for feature selection purposes
    x_train_tmp = normalize(x_train.to_numpy(), output_dir)
    model = feature_importance(x_train_tmp, y_train)
    x_train, x_test = feature_selection(x_train.to_numpy(), x_test.to_numpy(), temp_x_test.columns, model, output_dir, numfeat=50)

    # normalize the 50 features of interest of x_train and x_test
    x_train, x_test = normalize(x_train, output_dir, x_test)
    return x_train, y_train, x_test, y_test


###############
# def read_file(file, dim):
#     df = pd.read_csv(file)
#     x = df.iloc[:, dim[0]:dim[1]]
#     y = df['label_1']
#     return x, y
#
#
# def filter_features(data, file, numfeat=50):
#     df = pd.read_csv(file)
#     indices = list(df.sort_values(by=['feature_importances'], ascending=False).index[0:numfeat])
#     return data[:, indices]
#
#
# def read_files(file_path, dim, feature_file):
#     files = glob.glob(file_path)
#     print(files)
#     for i in range(len(files)):
#         if i == 0:
#             x, y = read_file(files[i], dim)
#         else:
#             temp_x, temp_y = read_file(files[i], dim)
#             x = pd.concat([x, temp_x], sort=False)
#             y = pd.concat([y, temp_y], sort=False)
#
#     x = filter_features(x.to_numpy(), feature_file)
#     return x, y
#
# def prepare_data():
#     train_path = '/home/jelle/Repositories/animalsounds/data/features/features_v7/features_sanct/*'
#     test_path = '/home/jelle/Repositories/animalsounds/data/features/test/sanaga/*'
#     output_dir = '/home/jelle/Repositories/animal-sounds/data/svm_results/prediction_results/sanctuary/'
#
#     feature_file = '../../data/svm_results/training_sanct/feature_importances.csv'
#     model_folder = '../../data/svm_results/training_sanct/_acoustic_model.sav'
#
#     dim = [5, -1]
#
#     x_train, y_train = read_files(train_path, dim, feature_file)
#     x_test, y_test = read_files(test_path, dim, feature_file)
#
#     x_train, x_test = normalize(x_train, x_test)
#
#     run_svm(x_train, y_train, x_test, y_test, output_dir, model_folder)
