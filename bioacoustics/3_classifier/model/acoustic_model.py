from abc import ABC, abstractmethod


class Acoustic_model(ABC):

    def __init__(self, X_train,y_train,X_test,y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def _train(self):
        pass

    def _predict(self):
        pass

    def save_model(self):
        pass

    def _load_model(self,file_name):
         pass

    def run_model(self):
        pass

    def save_results(self, file_name):
        pass

    def apply_model(self,file_name):
        pass


