from model.acoustic_model import Acoustic_model

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

import pickle

class RF_model(Acoustic_model):

    def __init__(self, X_train,y_train,X_test,y_test):
        super(RF_model, self).__init__(X_train,y_train,X_test,y_test)
        self.acoustic_model = None
        self.predicts = None


    def _train(self):
        parameters = {'n_estimators':(10,100,500)}

        self.acoustic_model = GridSearchCV(RandomForestClassifier(random_state=10), parameters)
        self.acoustic_model.fit(self.X_train,self.y_train)

    def _predict(self):
        self.predicts = self.acoustic_model.predict(self.X_test)

    def save_model(self, file_name):
        filename = file_name+'_acoustic_model.sav'
        pickle.dump(self.acoustic_model, open(filename, 'wb'))

    def load_model(self, file_name):
        # load the model from disk
        self.acoustic_model = pickle.load(open(file_name, 'rb'))

    def run_model(self):
        self._train()
        self._predict()

    def save_results(self,file_name):

        # save predictions
        with open(file_name+'_predictions.txt', "wb") as outfile:
            np.savetxt(outfile, self.predicts, fmt="%s")

        # save hyper-parameters / cross_validation results
        pd.DataFrame(self.acoustic_model.cv_results_).to_csv(file_name+'_gridSearch_results.csv')

        # save model
        self.save_model(file_name)

        # save y_test
        self.y_test.to_csv(file_name+'_y_test.csv', index=False)




