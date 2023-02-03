"""Script of a base class for acoustic models"""
import os
from abc import ABC
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras.metrics import Recall


class AcousticModel(ABC):
    """Base class for creating acoustic models"""

    def __init__(self):
        self.acoustic_model = None
        self.predicts = None

    def _make_cnn_model(self):
        """Make cnn model"""

    def _compile(self, learning_rate):
        """Compile acoustic model
        Parameters
        ----------
        learning_rate: float
                Learning rate for adam optimizer
        """
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate) #, decay=0.001

        # Compile the model
        self.acoustic_model.compile(
            loss="binary_crossentropy", #"categorical_crossentropy"
            metrics=['accuracy'], #Recall()
            optimizer=optimizer
        )

        # Display model architecture summary
        self.acoustic_model.summary()

    def _train(
        self,
        x_train,
        y_train,
        x_test=None,
        y_test=None,
        file_path=None,
        epochs=5,
        batch_size=32

    ):
        """Train acoustic model
                Parameters
                ----------
                x_train: pandas.DataFrame
                        A dataframe of training data
                y_train: pandas.DataFrame
                        A dataframe of data labels
                x_test: pandas.DataFrame
                        A dataframe of testing data
                y_test: pandas.DataFrame
                        A dataframe of testing labels
                epochs: int
                        The number of epochs
                batch_size: int
                        The batch size
                file_path: str
                        file path to save the trained model
        """

    def _predict(self, x_test):
        """Apply the Acoustic model on x_test
                Parameters
                ----------
                x_test: pandas.DataFrame
                        A dataframe of testing data
        """
        self.predicts = self.acoustic_model.predict(x_test)
        print("prediction is done!")

    def _load_model(self, file_path, dl_model):
        """Load the model from the given file path
        Parameters
        ----------
        file_path: str
                file path of the trained model
        dl_model: bool
                indicates if it is a deep-learning model or not
        """
        if dl_model:
            self.acoustic_model = load_model(file_path)
        else:
            self.acoustic_model = pickle.load(open(file_path, "rb"))

    def make_model(
        self,
        init_mode="glorot_uniform",
        dropout_rate=0.2,
        weight_constraint=3,
        learning_rate=0.001,
        compile_model=True
    ):
        """Make a model with the given configuration and compile it in case of
        compile_model=True
                Parameters
                ----------
                init_mode: str
                        statistical distribution or function to use for
                        initialising the weights.
                dropout_rate: float
                        drop-out rate
                weight_constraint: float
                        weight constraint
                learning_rate: float
                        learning rate for the optimizer, in our case adam.
                compile_model: bool
                        indicates if the model needs to be compiled or not
        """
        self._make_cnn_model(init_mode, dropout_rate, weight_constraint)
        if compile_model:
            self._compile(learning_rate)
        return self.acoustic_model

    def apply_model(
        self, x_train, y_train, x_test, y_test, file_path, epochs, batch_size
    ):
        """Train a model and make a prediction on test dataset
                        Parameters
                        ----------
                        x_train: pandas.DataFrame
                                A dataframe of training data
                        y_train: pandas.DataFrame
                                A dataframe of data labels
                        x_test: pandas.DataFrame
                                A dataframe of testing data
                        y_test: pandas.DataFrame
                                A dataframe of testing labels
                        file_path: str
                                file path to save the trained model
                        epochs: int
                                The number of epochs
                        batch_size: int
                                The batch size

        """
        self._train(x_train, y_train, x_test, y_test, file_path,
                    epochs, batch_size)
        self._predict(x_test)

    def predict_model(self, x_test, file_path, dl_model):

        """Load a trained model and make a prediction
        Parameters
        ----------
        x_test: pandas.DataFrame
            A dataframe of testing data
        file_path: str
            file path of the trained model
        dl_model: bool
            indicates if the trained model is a deep learning model or not
        """
        self._load_model(file_path, dl_model)
        self._predict(x_test)

    def save_results(
        self, y_test, file_path, predicts_only=False
    ):
        """Save predictions of the test dataset
        Parameters
        ----------
        y_test: pandas.DataFrame
             A dataframe of testing labels
        file_path: str
            file path to save the predictions
        predicts_only: bool
            Indicates if tests labels need to be saved, Default True
        """

        with open(file_path + "_predictions.txt", "wb") as outfile:
            np.savetxt(outfile, self.predicts, fmt="%s")

        # when a trained model is applied on un-labeled dataset,
        # only predictions need to be saved
        if predicts_only:
            return

        # save y_test
        pd.DataFrame(y_test).to_csv(file_path + "_y_test.csv", index=False)

    def plot_measures(self, history, file_path, title=""):
        """Summarize history for recall
                Parameters
                ----------
                history: tf.keras.callbacks.History
                     model history
                file_path: str
                    file path to save the image of the graph
                title: str
                    Title of the graph
        """
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        fp_loss = os.path.join(file_path, 'loss.png')
        plt.savefig(fp_loss)

        # convert the history.history dict to a pandas DataFrame:
        hist_df = pd.DataFrame(history.history)
        hist_csv_file = os.path.join(file_path, "history.csv")
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)
