import numpy as np

from BayesianClassification.src.classifiers.BinaryClassifier import BinaryClassifier


def estimate_per_class(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    mean = X.mean(axis=0, keepdims=True)
    n = X.shape[0]
    S = np.cov(X, rowvar=False)
    return mean, S, n


class QDA(BinaryClassifier):

    def __init__(self):
        super().__init__()
        self.prior = None
        self.n1 = None
        self.n0 = None
        self.m1 = None
        self.m0 = None
        self.S0 = None
        self.S1 = None
        self.S0_inv = None
        self.S1_inv = None
        self.det_S0 = None
        self.det_S1 = None

    def __delta_10(self, X):
        delta_0 = -0.5 * np.log(self.det_S0) - 0.5 * np.sum((X - self.m0) @ self.S0_inv * (X - self.m0),
                                                            axis=1) + np.log(
            1 - self.prior)
        delta_1 = -0.5 * np.log(self.det_S1) - 0.5 * np.sum((X - self.m1) @ self.S1_inv * (X - self.m1),
                                                            axis=1) + np.log(
            self.prior)
        return delta_1 - delta_0

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.prior = y.mean()
        self.m0, self.S0, self.n0 = estimate_per_class(X[y == 0])
        self.m1, self.S1, self.n1 = estimate_per_class(X[y == 1])
        self.S0_inv = np.linalg.inv(self.S0)
        self.S1_inv = np.linalg.inv(self.S1)
        self.det_S0 = np.linalg.det(self.S0)
        self.det_S1 = np.linalg.det(self.S1)

    def predict_proba(self, X_test: np.ndarray):
        return 1 / (1 + np.exp(-self.__delta_10(X_test)))

    def predict(self, X_test: np.ndarray):
        return (self.__delta_10(X_test) > 0).astype(int)

    def get_params(self):
        return [self.n0, self.n1, self.m0, self.m1, self.S0, self.S1]
