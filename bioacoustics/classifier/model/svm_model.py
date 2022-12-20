from .acoustic_model import AcousticModel
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pickle


class SVM_model(AcousticModel):
    def __init__(self, *args):
        super(SVM_model, self).__init__()
        self.predicts = None

    def _train(self, X_train, y_train, X_test, y_test, file_path):
        parameters = [
            {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
            {"kernel": ["linear"], "C": [1, 10, 100, 1000]},
        ]

        self.acoustic_model = GridSearchCV(
            SVC(), parameters, scoring="recall_macro", n_jobs=10
        )
        self.acoustic_model.fit(X_train, y_train)

        print("Best parameter (UAR=%0.3f):" % self.acoustic_model.best_score_)
        print(self.acoustic_model.best_params_)
        self._save_model(file_path)

    def _save_model(self, file_path):
        filename = file_path + "svm_model.sav"
        pickle.dump(self.acoustic_model, open(filename, "wb"))
