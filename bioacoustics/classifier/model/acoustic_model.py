"""Script of a base class for acoustic models"""

import os
from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import json
import warnings


class AcousticModel(ABC):
    def __init__(self, output_dir: str, model_dir: str | None = None):
        self.output_dir = output_dir
        self.model_dir = model_dir or output_dir
        self.acoustic_model = None
        self.threshold = 0.5

    @abstractmethod
    def fit(
            self,
            x_train,
            y_train,
            x_val=None,
            y_val=None,
            **kwargs
    ):
        """Train acoustic model

        Parameters
        ----------
        x_train: pandas.DataFrame
            A dataframe of training data
        y_train: pandas.DataFrame
            A dataframe of data labels
        x_val: pandas.DataFrame
            A dataframe of testing data
        y_val: pandas.DataFrame
            A dataframe of testing labels
        """

    @abstractmethod
    def _save_model(self):
        """Load the model from the given self.output_dir
        """

    @abstractmethod
    def _load_model(self):
        """Load the model from self.output_dir
        """

    @abstractmethod
    def _predict(self, x, **kwargs):
        """Apply the Acoustic model on x_test

        Parameters
        ----------
        x: pandas.DataFrame
            A dataframe of testing data
        threshold: float
            threshold to set prediction
        """

    def _save_threshold(self, threshold):
        path = os.path.join(self.output_dir, "threshold.json")
        with open(path, "w") as f:
            json.dump({"threshold": float(threshold)}, f)

    def _load_threshold(self, default=0.5):
        path = os.path.join(self.model_dir, "threshold.json")
        if not os.path.exists(path):
            warnings.warn(
                "Threshold file not found. Using default threshold.",
                RuntimeWarning
            )
            self.threshold = default
            return self.threshold

        with open(path) as f:
            self.threshold = json.load(f).get("threshold")

    @abstractmethod
    def evaluate_model(self, x, y, **kwargs):
        """
        Evaluate model performance on labeled data.

        Must be implemented in the derived class.

        Parameters
        ----------
        x: np.ndarray or pandas.DataFrame
            Input features
        y: np.ndarray or pandas.DataFrame
            Ground truth labels

        Returns
        -------
        dict: metrics like {'accuracy': ..., 'loss': ...}
        """

    def predict_model(self, x, **kwargs):
        """Load a trained model and make a prediction for the labels

        Parameters
        ----------
        x: pandas.DataFrame
            A dataframe for the prediction
        """
        self._load_model()
        predict_result = self._predict(x, **kwargs)
        return predict_result

    def evaluate(self, x, y, **kwargs):
        """Load a trained model and make a prediction for the labels

        Parameters
        ----------
        x: pandas.DataFrame
            A dataframe for the prediction
        y: pandas.DataFrame
            A dataframe including labels or None

        """
        self._load_model()
        result1, result2 = self.evaluate_model(x, y, **kwargs)
        return result1, result2

    def _plot_measures(self, history,  title=""):
        """Plot loss curves and save CSV."""

        plt.figure()
        plt.plot(history["train_loss"])
        plt.plot(history["val_loss"])
        plt.title("Model Loss " + title)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(["train", "val"])
        plt.savefig(os.path.join(self.output_dir, "loss.png"))
        plt.close()

        pd.DataFrame(history).to_csv(os.path.join(self.output_dir, "history.csv"))

    @abstractmethod
    def save_evaluation(
            self,
            probs,
            metrics=None,
            predicts=None,
            y_true=None,
            split="",
    ):
        """Save both predictions and metrics for a dataset split."""
