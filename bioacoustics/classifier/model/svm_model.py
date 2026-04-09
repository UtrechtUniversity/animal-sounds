"""SVM model class."""

import pickle
import numpy as np
import pandas as pd
from bioacoustics.classifier.model.acoustic_model import AcousticModel
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import json


class SVM_model(AcousticModel):
    """SVM model.

    Parameters
    ----------
    args : list
        List of arguments.
    """

    def __init__(self, output_dir, model_dir=None):
        super().__init__(output_dir=output_dir, model_dir=model_dir)
        self.acoustic_model = None

    def fit(self,
            x_train,
            y_train,
            x_val=None,
            y_val=None,
            *args,
            **kwargs):
        parameters = [
            {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
            {"kernel": ["linear"], "C": [1, 10, 100, 1000]},
        ]

        self.acoustic_model = GridSearchCV(
            SVC(), parameters, scoring="recall_macro", n_jobs=10
        )
        self.acoustic_model.fit(x_train, y_train)

        print(f"Best parameter (UAR={self.acoustic_model.best_score_}")
        print(self.acoustic_model.best_params_)
        self._save_model()

    def _save_model(self):
        filename = self.output_dir + "svm_model.sav"
        with open(filename, "wb") as file:
            pickle.dump(self.acoustic_model, file)

    def _load_model(self):
        """Load the model from the given file path
        """
        filename = self.model_dir + "svm_model.sav"
        with open(filename, "rb") as f:
            self.acoustic_model = pickle.load(f)

    def evaluate_model(self, x, y, **kwargs):
        preds = self.acoustic_model.predict(x)
        acc = np.mean(preds == y)

        print(f"Evaluation — accuracy={acc:.4f}")
        return acc, preds

    def _predict(self, x, **kwargs):
        """Apply the Acoustic model on x

        Parameters
        ----------
        x: pandas.DataFrame
            A dataframe of testing data
        """
        predicts = self.acoustic_model.predict(x)
        print("prediction is done!")
        return predicts

    def save_evaluation(
            self,
            probs,
            metrics=None,
            predicts=None,
            y_true=None,
            split="",
    ):
        """Save both predictions and metrics for a dataset split."""

        with open(self.output_dir + split + "_predictions.txt", "wb") as outfile:
            np.savetxt(outfile, probs, fmt="%s")

        # when a trained model is applied on un-labeled dataset,
        # only predictions need to be saved
        if y_true is not None:
            pd.DataFrame(y_true).to_csv(self.output_dir + split + "_y_test.csv", index=False)

        metrics_path = f"{self.output_dir}{split}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

