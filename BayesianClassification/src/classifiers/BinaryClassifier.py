from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, recall_score, precision_score

import numpy as np


class BinaryClassifier(ABC):

    @abstractmethod
    def fit(self,
            X: np.ndarray,
            y: np.ndarray):
        pass

    @abstractmethod
    def predict_proba(self, X_test: np.ndarray):
        pass

    @abstractmethod
    def predict(self, X_test: np.ndarray):
        pass

    @abstractmethod
    def get_params(self):
        pass

    def fit_predict_evaluate(self, X, y, X_test, y_test, metric):
        self.fit(X, y)
        y_hat = self.predict(X_test)
        match metric:
            case "accuracy":
                return accuracy_score(y_test, y_hat)
            case "recall":
                return recall_score(y_test, y_hat)
            case "precision":
                return precision_score(y_test, y_hat)
            case _:
                raise NotImplementedError
