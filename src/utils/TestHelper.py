import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from src.classifiers.BinaryClassifier import BinaryClassifier
from src.classifiers.LDA import LDA
from src.classifiers.NaiveBayes import NaiveBayes
from src.classifiers.QDA import QDA
from src.utils.DatasetGenerator import DatasetGenerator


def fit_predict_plot(classifier: BinaryClassifier, X, y, color):
    classifier.fit(X, y)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    z = classifier.predict_proba(grid).reshape(xx.shape)
    plt.contour(xx, yy, z, levels=[0.5], colors=color, linestyles='dashed')


class TestHelper:
    def __init__(self, metric="accuracy"):
        self.ylim = None
        self.generator = DatasetGenerator()
        if metric not in ["accuracy", "recall", "precision"]:
            raise NotImplementedError
        self.metric = metric

    def __test(self, X, y, times, test_size):
        results = [[None, None, None] for _ in range(times)]
        classifiers = [LDA(), QDA(), NaiveBayes()]
        for i in range(times):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            for j in range(len(classifiers)):
                score = classifiers[j].fit_predict_evaluate(X_train, y_train, X_test, y_test, self.metric)
                results[i][j] = score
        return np.array(results)

    def test_scheme_1(self, a, times, test_size=0.2):
        X, y = self.generator.schema_1(a)
        self.__make_boxplot(self.__test(X, y, times, test_size), test_size, a)

    def test_scheme_2(self, a, rho, times, test_size=0.2):
        X, y = self.generator.schema_2(a, rho)
        self.__make_boxplot(self.__test(X, y, times, test_size), test_size, a, rho)

    def test_real_data(self, X, y, times, test_size=0.2):
        self.__make_boxplot(self.__test(X, y, times, test_size), test_size)

    def __make_boxplot(self, scores, test_size, a=None, rho=None):
        if rho:
            title = f"Comparison for schema 2: a={a}, rho={rho}. Tested {scores.shape[0]} times with test size {test_size}"
        elif a:
            title = f"Comparison for schema 1: a={a}. Tested {scores.shape[0]} times with test size {test_size}"
        else:
            title = f"Comparison for real data. Tested {scores.shape[0]} times with test size {test_size}"

        data = [scores[:, 0], scores[:, 1], scores[:, 2]]
        plt.figure(figsize=(9, 6))
        sns.boxplot(data=data)
        plt.title(title)
        plt.ylabel(self.metric)
        if self.ylim is not None:
            plt.ylim(self.ylim)
        plt.xticks([0, 1, 2], ['LDA', 'QDA', 'NaiveBayes'])
        plt.show()

    def plot_boundaries(self, a=None, rho=None, X=None, y=None, test_size=0.2):
        if rho:
            X, y = self.generator.schema_2(a, rho)
        elif a:
            X, y = self.generator.schema_1(a)
        else:
            assert X is not None
            assert X.shape[1] == 2
            assert y is not None

        X1 = X[y == 1]
        X0 = X[y == 0]
        plt.figure(figsize=(12, 9))
        plt.scatter(X0[:, 0], X0[:, 1], c='blue', marker='d', s=7, label='Class 0')
        plt.scatter(X1[:, 0], X1[:, 1], c='orange', marker='P', s=7, label='Class 1')

        fit_predict_plot(LDA(), X, y, 'green')
        fit_predict_plot(QDA(), X, y, 'purple')

        plt.legend()
        plt.title("LDA and QDA decision boundaries comparison")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()
