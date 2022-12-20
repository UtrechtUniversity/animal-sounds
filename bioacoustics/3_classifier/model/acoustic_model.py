import os
from abc import ABC
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras.metrics import Recall
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


class AcousticModel(ABC):
    """Train a model and make a prediction on test dataset
            Parameters
            ----------
            file_name: str
                file path to save the trained model
            X_train: bool
                Indicates if output should be saved

            Returns
            -------
            np.ndarrary:
                Padded audio file.
    """
    def __init__(self):
        self.acoustic_model = None

    def _make_cnn_model(self):
        pass
    # def _compile(self):
    #     pass

    def _compile(self, learning_rate):
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate) #0.0001

        # Compile the model
        self.acoustic_model.compile(loss='categorical_crossentropy', metrics=[Recall()], optimizer=optimizer)# optimizer='adam')  # 'accuracy'

        # Display model architecture summary
        self.acoustic_model.summary()

    def _train(self, X_train,y_train,epochs, batch_size,X_test=None,y_test=None, file_name=None):
        pass

    def _predict(self, X_test):
        """Apply the Acoustic model on X_test """
        self.predicts = self.acoustic_model.predict(X_test)
        print('prediction is done!')

    def _load_model(self, file_path, dl_model):
        """Load the model from the given file path
        Parameters
        ----------
        file_path: str
                file path of the trained model
        """
        if dl_model:
            self.acoustic_model = load_model(file_path)
        else:
            self.acoustic_model = pickle.load(open(file_path, 'rb'))

    def make_model(self, init_mode="glorot_uniform", dropout_rate=0.2, weight_constraint=3, learning_rate =0.001,
                   compile_model=True):
        print('inside make_model')
        self._make_cnn_model(init_mode, dropout_rate,weight_constraint)
        if compile_model:
            self._compile(learning_rate)
        return self.acoustic_model

    def apply_model(self, X_train, y_train, X_test, y_test, file_path, epochs, batch_size):
        """Train a model and make a prediction on test dataset
        Parameters
        ----------
        file_path: str
            file path to save the trained model
        X_train: pandas dataframe
            Indicates if output should be saved

        Returns
        -------
        np.ndarrary:
            Padded audio file.
        """

        self._train(X_train, y_train, X_test, y_test, file_path, epochs, batch_size)
        # self.evaluate_result = self.acoustic_model.evaluate(X_test, y_test, batch_size=32)
        # print(self.evaluate_result)
        self._predict(X_test)

    def predict_model(self, X_test, file_path, dl_model):
        """Load a trained model and make a prediction
            Parameters
            ----------
            file_path: str
                file path of the trained model
        """
        self._load_model(file_path,dl_model)
        self._predict(X_test)

    def save_results(self, y_test, file_path, predicts_only=False):#, cv_results= False):
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
        # if cv_results:
        #     pd.DataFrame(self.acoustic_model.cv_results_).to_csv(file_path+'_gridSearch_results.csv')

        # save y_test
        pd.DataFrame(y_test).to_csv(file_path+'_y_test.csv', index=False)


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






