import os
from abc import ABC, abstractmethod
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class AcousticModel(ABC):

    def __init__(self):
        self.acoustic_model = None
        self.evaluate_result = None

    def _train(self, X_train,y_train,X_test,y_test, file_name):
        pass

    def _predict(self, X_test):
        """Apply the Acoustic model on X_test """
        self.predicts = self.acoustic_model.predict(X_test)
        print('prediction is done!')

    def _load_model(self, file_path):
        """Load the model from the given file path
            Parameters
            ----------
            file_path: str
                    file path of the trained model
        """
        self.acoustic_model = load_model(file_path)

    def apply_model(self, X_train,y_train,X_test,y_test, file_name):
        """Train a model and make a prediction on test dataset
            Parameters
            ----------
            file_name: str
                    file path to save the trained model
        """

        self._train(X_train,y_train,X_test,y_test,file_name)
        self.evaluate_result = self.acoustic_model.evaluate(X_test, y_test, batch_size=32)
        print(self.evaluate_result)
        self._predict(X_test)

    def predict_model(self, X_test, file_path):
        """Load a trained model and make a prediction
            Parameters
            ----------
            file_path: str
                file path of the trained model
        """
        self._load_model(file_path)
        self._predict(X_test)

    def evaluate_model(self, X_test, y_test, file_path):
        """Load a trained model and make an evaluation on test dataset
            Parameters
            ----------
            file_path: str
                file path of the trained model
        """
        self._load_model(file_path)
        self.evaluate_result = self.acoustic_model.evaluate(X_test, y_test, batch_size=32)
        print("test loss, test acc:", self.evaluate_result)

    def save_results(self, y_test, file_path, predicts_only=False):
        """Save predictions on test dataset
                Parameters
                ----------
                file_path: str
                    file path to save the predictions
                predicts_only: bool
                    Indicates if labels need to be saved, Default True
        """

        with open(file_path+'_predictions.txt', "wb") as outfile:
            np.savetxt(outfile, self.predicts, fmt="%s")

        # when a trained model is applied on un-labeled dataset, only predictions need to be saved
        if predicts_only:
            return

        # # save hyper-parameters / cross_validation results
        # pd.DataFrame(self.acoustic_model.cv_results_).to_csv(file_name+'_gridSearch_results.csv')

        # save y_test
        pd.DataFrame(y_test).to_csv(file_path+'_y_test.csv', index=False)

#   check if it is used
#     def _save_model(self, file_path):
#         """Save the trained model in the given file path
#             Parameters
#             ----------
#             file_path: str
#                     file path to save the trained model
#         """
#         self.acoustic_model.save(file_path)
    def plot_measures(self, history, file_path, title=''):
        # summarize history for recall
        plt.plot(history.history['recall'])
        plt.plot(history.history['val_recall'])
        plt.title('model recall '+title)
        plt.ylim([0,1])
        plt.ylabel('recall')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        fp_recall = os.path.join(file_path,'recall.png')
        plt.savefig(fp_recall)
        # # summarize history for loss
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'val'], loc='upper left')
        # fp_loss = os.path.join(file_path, 'loss.png')
        # plt.savefig(fp_loss)






