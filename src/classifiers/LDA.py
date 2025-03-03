import numpy as np

from src.classifiers.BinaryClassifier import BinaryClassifier


def estimate_per_class(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    mean = X.mean(axis=0, keepdims=True)
    n = X.shape[0]
    S = np.cov(X, rowvar=False)
    return mean, S, n


class LDA(BinaryClassifier):

    def __init__(self):
        super().__init__()
        self.prior = None
        self.n1 = None
        self.n0 = None
        self.m1 = None
        self.m0 = None
        self.W = None
        self.W_inv = None

    def __delta_10(self, X):
        m_diff = (self.m1 - self.m0).reshape(-1, 1)
        m_sum = (self.m1 + self.m0).reshape(-1, 1)

        return (X @ self.W_inv @ m_diff) - \
            0.5 * (m_diff.T @ self.W_inv @ m_sum) + \
            np.log(self.n1 / self.n0)
        # return (X @ self.W_inv @ (self.m1 - self.m0).T) - \
        #     0.5 * (self.m1 - self.m0).T @ self.W_inv @ (self.m1 + self.m0) + \
        #     np.log(self.n1 / self.n0)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.prior = y.mean()
        self.m0, s0, self.n0 = estimate_per_class(X[y == 0])
        self.m1, s1, self.n1 = estimate_per_class(X[y == 1])
        self.W = ((self.n0 - 1) * s0 + (self.n1 - 1) * s1) / (self.n0 + self.n1 - 2)
        self.W_inv = np.linalg.inv(self.W)

    def predict_proba_M(self, X_test: np.ndarray):
        X_1 = X_test - self.m1
        X_0 = X_test - self.m0

        prob0 = X_0 @ self.W_inv
        prob0 = np.sum(prob0 * X_0, axis=1)
        prob0 = (1 - self.prior) * np.exp((-0.5) * prob0)

        prob1 = X_1 @ self.W_inv
        prob1 = np.sum(prob1 * X_1, axis=1)
        prob1 = self.prior * np.exp((-0.5) * prob1)

        return np.divide(prob1, (prob1 + prob0)).T

    def predict_proba(self, X_test: np.ndarray):
        return 1 / (1 + np.exp(-self.__delta_10(X_test)))

    def predict(self, X_test: np.ndarray):
        return (self.__delta_10(X_test) > 0).astype(int)

    def get_params(self):
        return [self.n0, self.n1, self.m0, self.m1, self.W]
