"""SVM model class."""

import pickle
from acoustic_model import AcousticModel
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


class SVM_model(AcousticModel):
    """SVM model.

    Parameters
    ----------
    args : list
        List of arguments.
    """

    def __init__(self, *args):
        super().__init__()
        self.predicts = None
        self.acoustic_model = None

    def _train(self, X_train, y_train, X_test, y_test, file_path):
        parameters = [
            {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
            {"kernel": ["linear"], "C": [1, 10, 100, 1000]},
        ]

        self.acoustic_model = GridSearchCV(
            SVC(), parameters, scoring="recall_macro", n_jobs=10
        )
        self.acoustic_model.fit(X_train, y_train)

        print(f"Best parameter (UAR={self.acoustic_model.best_score_}")
        print(self.acoustic_model.best_params_)
        self._save_model(file_path)

    def _save_model(self, file_path):
        filename = file_path + "svm_model.sav"
        with open(filename, "wb") as file:
            pickle.dump(self.acoustic_model, file)
