import pandas as pd
import numpy as np
import glob
#from torch import FloatTensor
#from torch import tensor
#from torch import int64
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from model.svm_model import SVM_model
#from model.elm_model import MLP
import sklearn.model_selection as model_selection
from sklearn.ensemble import ExtraTreesClassifier
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
#from torch.utils.data import Dataset
#from torch.utils.data import DataLoader
#from torchvision import transforms
#import pytorch_lightning as pl


# custom dataset
class customDataset(Dataset):
    def __init__(self, features, labels=None, transforms=None):
        self.X = features
        self.y = labels
        self.transforms = transforms

    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        data = self.X[i, :]
        # data = data.astype(np.uint8)

        if self.transforms:
            data = self.transforms(data)
        else:
            data = FloatTensor(data)

        if self.y is not None:
            return (data, tensor(self.y[i], dtype=int64))
        else:
            return data


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
    # recursive_features(x_train, y_train, temp_x_test.columns, output_dir)

    model = feature_importance(x_train, y_train)
    x_train, x_test = feature_selection(x_train, x_test, temp_x_test.columns, model, output_dir, numfeat=50)
    if ml_method == 'svm':
        run_svm(x_train, y_train, x_test, y_test, output_dir)
    elif ml_method == 'nn':
        # define transforms
        le = LabelEncoder()
        le.fit(y_train)
        train_data = customDataset(x_train, le.transform(y_train))
        test_data = customDataset(x_test, le.transform(y_test))

        pl.seed_everything(42)
        mlp = MLP()
        trainer = pl.Trainer(auto_scale_batch_size='power', gpus=1, deterministic=True, max_epochs=5)
        trainer.fit(mlp, DataLoader(train_data))
        result = trainer.test(test_dataloaders=DataLoader(test_data))
        print(result)


if __name__ == "__main__":
    ml_method = 'svm'  # 'nn' or 'svm'
    main(ml_method)
