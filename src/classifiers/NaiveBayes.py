import numpy as np

from src.classifiers.BinaryClassifier import BinaryClassifier


def estimate_per_class(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0, keepdims=True)
    var = np.var(X, axis=0)
    return mean, var


def gaussian_density(X, mean, var):
    coef = 1.0 / np.sqrt(2 * np.pi * var)
    exponent = np.exp(-((X - mean) ** 2) / (2 * var))
    return coef * exponent


class NaiveBayes(BinaryClassifier):

    def __init__(self):
        super().__init__()
        self.prior = None
        self.m1 = None
        self.m0 = None
        self.var1 = None
        self.var0 = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.prior = y.mean()
        self.m0, self.var0 = estimate_per_class(X[y == 0])
        self.m1, self.var1 = estimate_per_class(X[y == 1])

    def predict_proba(self, X_test: np.ndarray):
        prob0 = np.prod(gaussian_density(X_test, self.m0, self.var0), axis=1) * (1 - self.prior)
        prob1 = np.prod(gaussian_density(X_test, self.m1, self.var1), axis=1) * self.prior

        return prob1 / (prob0 + prob1)

    def predict(self, X_test: np.ndarray):
        return (self.predict_proba(X_test) > 0.5).astype(int)

    def get_params(self):
        return [self.m0, self.m1, self.var0, self.var1, self.prior]
