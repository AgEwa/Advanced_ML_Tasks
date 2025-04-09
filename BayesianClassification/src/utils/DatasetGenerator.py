import numpy as np


class DatasetGenerator:
    def __init__(self, n=1000, prior=0.5, p=2):
        self.n = n
        self.prior = prior
        self.p = p

    def schema_1(self, a):
        y_start = np.random.binomial(n=1, p=self.prior, size=self.n)
        X0 = np.random.normal(0, 1, size=(sum(y_start == 0), self.p))
        X1 = np.random.normal(a, 1, size=(sum(y_start == 1), self.p))
        X = np.vstack((X0, X1))
        y = np.concatenate((np.zeros(len(X0)), np.ones(len(X1))))
        return X, y

    def schema_2(self, a, rho):
        cov0 = [[1, rho], [rho, 1]]
        cov1 = [[1, -rho], [-rho, 1]]
        y_start = np.random.binomial(n=1, p=self.prior, size=self.n)
        X0 = np.random.multivariate_normal([0, 0], cov0, sum(y_start == 0))
        X1 = np.random.multivariate_normal([a, a], cov1, sum(y_start == 1))
        X = np.vstack((X0, X1))
        y = np.concatenate((np.zeros(len(X0)), np.ones(len(X1))))
        return X, y
