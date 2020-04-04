from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class MOELScheme(ABC, BaseEstimator):

    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def cross_val_score(self, X, y, num_fold):
        pass

    @abstractmethod
    def show_pareto(self):
        pass

    @abstractmethod
    def show_model(self, position):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass


class MOEL_FRBC(MOELScheme, ClassifierMixin):

    def __init__(self, classifiers=None):
        if classifiers is None:
            classifiers = []
        self.classifiers = classifiers

    def predict(self, X, position='first'):
        n_classifiers = len(self.classifiers)
        if n_classifiers == 0:
            return None
        else:
            index = {'first': 0, 'median': int(n_classifiers/2), 'last': -1}
            return self.classifiers[index[position.lower()]].predict(X)

    def score(self, X, y, sample_weight=None):
        n_classifiers = len(self.classifiers)
        if n_classifiers > 1:
            indexes = {0, int(n_classifiers / 2), n_classifiers - 1}
            for index in indexes:
                clf = self.classifiers[index]
                accuracy = clf.accuracy()
                complexity = clf.trl()
                print(accuracy, complexity)



class MOEL_FRBR(MOELScheme, RegressorMixin):

    def __init__(self, regressors=None):
        if regressors is None:
            regressors = []
        self.regressors = regressors

    def predict(self, X, position='first'):
        n_regressors = len(self.regressors)
        if n_regressors == 0:
            return None
        elif n_regressors == 1:
            return self.regressors[0].predict()
        else:
            index = {'first': 0, 'median': int(n_regressors / 2), 'last': n_regressors - 1}
            return self.regressors[index[position]].predict(X)

    def score(self, X, y, sample_weight=None):
        n_regressors = len(self.regressors)
        if n_regressors > 1:
            indexes = {0, int(n_regressors / 2), n_regressors - 1}
            for index in indexes:
                rgr = self.regressors[index]
                accuracy = rgr.accuracy()
                complexity = rgr.trl()
                print(accuracy, complexity)
