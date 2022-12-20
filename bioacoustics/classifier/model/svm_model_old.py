from .acoustic_model import AcousticModel
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
import pickle


class SVM_model(AcousticModel):

    def __init__(self, X_train, y_train, X_test, y_test):
        super(SVM_model, self).__init__(X_train, y_train, X_test, y_test)
        self.acoustic_model = None
        self.predicts = None
        self.predict_probs = None

    def _train(self):
        # base on https://stackoverflow.com/questions/62603509/using-pickle-to-load-random-forest-model-gives-the-wrong-prediction
        parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                       'C': [1, 10, 100, 1000]},
                      {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

        # {'svm__kernel': ['linear', 'rbf'], 'svm__C': (1, 10, 20)}

        #pipe = Pipeline([('svm', SVC(probability=False))])
        # ('feature_selection', SelectFromModel(LinearSVC(penalty="l1")))

        # predict_prob
        # self._acoustic_model = GridSearchCV(pipe, parameters)  # scoring='recall_macro'
        # self._acoustic_model.fit(self.X_train,self.y_train)
        #
        # print("Best parameter (CV score=%0.3f):" % self._acoustic_model.best_score_)
        # print(self._acoustic_model.best_params_)
        #
        # # we use calibratedClassifierCV to get predict_prob
        # self.acoustic_model = CalibratedClassifierCV(self._acoustic_model.best_estimator_)
        # self.acoustic_model.fit(self.X_train, self.y_train)

        self.acoustic_model = GridSearchCV(SVC(), parameters, scoring='recall_macro', n_jobs=10)
        self.acoustic_model.fit(self.X_train, self.y_train)

        print("Best parameter (UAR=%0.3f):" % self.acoustic_model.best_score_)
        print(self.acoustic_model.best_params_)

    def _predict(self):
        self.predicts = self.acoustic_model.predict(self.X_test)

        # predict_prob
        # self.predict_probs = self.acoustic_model.predict_proba(self.X_test)

        print('prediction is done!')

    def save_model(self, file_name):
        filename = file_name + '_acoustic_model.sav'
        pickle.dump(self.acoustic_model, open(filename, 'wb'))

    def _load_model(self, file_name):
        # load the model from disk
        self.acoustic_model = pickle.load(open(file_name, 'rb'))

    def run_model(self):
        self._train()
        self._predict()

    def save_results(self, file_name, predicts_only=False):
        # save predictions
        with open(file_name + '_predictions.txt', "wb") as outfile:
            np.savetxt(outfile, self.predicts, fmt="%s")

        # save prediction probs
        # pd.DataFrame(self.predict_probs).to_csv(file_name + '_prediction_probs.csv', index=False)

        # when a trained model is applied, only predictions need to be saved
        if predicts_only:
            return

        # save hyper-parameters / cross_validation results
        pd.DataFrame(self.acoustic_model.cv_results_).to_csv(file_name + '_gridSearch_results.csv')

        # save model
        self.save_model(file_name)

        # save y_test
        self.y_test.to_csv(file_name + '_y_test.csv', index=False)

    def apply_model(self, file_name):
        self._load_model(file_name)
        self._predict()
