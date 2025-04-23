import numpy as np
from sklearn import base
from sklearn import tree


class AdaBoost:
    def __init__(self, base_learner=None, n_classifiers=50):
        self.base_learner = base_learner or tree.DecisionTreeClassifier(max_depth=1)
        self.n_classifiers = n_classifiers
        self.models = []
        self.betas = []
        self.classes_ = None

    def fit(self, X, y):
        n = len(y)
        w = np.ones(n) / n
        self.classes_ = np.unique(y)

        for _ in range(self.n_classifiers):
            clf = base.clone(self.base_learner)
            clf.fit(X, y, sample_weight=w)
            pred = clf.predict(X)

            incorrect = (pred != y)
            epsilon = np.sum(w * incorrect)

            if epsilon == 1:
                epsilon = 1 - 1e-10  # avoid dividing by zero later

            beta = epsilon / (1 - epsilon)
            self.models.append(clf)
            self.betas.append(beta)

            w *= np.where(incorrect, 1, beta)
            w /= np.sum(w)

    def predict(self, X):
        if self.classes_ is None: raise Exception("No classes: Model must be fitted first")
        weighted_votes = np.zeros((X.shape[0], len(self.classes_)))

        for model, beta in zip(self.models, self.betas):
            preds = model.predict(X)
            for i, pred in enumerate(preds):
                class_index = np.where(self.classes_ == pred)[0][0]
                weighted_votes[i, class_index] += np.log(1 / beta)

        return self.classes_[np.argmax(weighted_votes, axis=1)]
