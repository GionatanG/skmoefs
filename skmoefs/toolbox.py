import numpy as np
import os
import time
import copy
import pickle
import random
import re
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl

from platypus import AdaptiveGridArchive

from skmoefs.moea import MPAES2_2
from skmoefs.moel import MOEL_FRBC
from skmoefs.rcs import RCSProblem, RCSVariator, RCSInitializer
from skmoefs.moea import MOEAGenerator, RandomSelector, NSGAIIS, NSGAIIIS, GDE3S, SPEA2S, IBEAS, MOEADS, EpsMOEAS

from sklearn.model_selection import KFold
from sklearn import metrics

milestones = [500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000, 75000, 100000]


def is_object_present(name):
    return os.path.isfile(name + '.obj')


def store_object(object, name):
    try:
        f = open(name + '.obj', 'wb')
        pickle.dump(object, f)
    except IOError:
        pass
    finally:
        f.close()


def load_object(name):
    try:
        f = open(name + '.obj', 'rb')
        return pickle.load(f)
    except IOError:
        pass
    finally:
        f.close()


def set_default_plot_style():
    mpl.style.use('classic')
    plt.rc('font', size=10)  # controls default text sizes
    plt.rc('axes', titlesize=14)  # fontsize of the axes title
    plt.rc('axes', labelsize=16)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=14)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=14)  # fontsize of the tick labels
    plt.rc('legend', fontsize=14)  # legend fontsize
    plt.rc('figure', titlesize=14)  # fontsize of the figure title


def load_dataset(name):
    """
    Load some predefined dataset (Format .dat)

    :param name: dataset name
    :return:
        X: NumPy matrix NxM (N: number of samples; M: number of features) representing input data
        y: Numpy vector Nx1 representing the output data
        attributes: range of values in the format [min, max] for each feature
        input: names of the features
        output: name of the outputs

    """
    attributes = []
    inputs = []
    outputs = []
    X = []
    y = []
    os.listdir('dataset')
    with open('dataset/' + name + '.dat', 'r') as f:
        line = f.readline()
        while line:
            if line.startswith("@"):
                txt = line.split()
                if txt[0] == "@attribute":
                    domain = re.search('(\[|\{)(.+)(\]|\})', line)
                    attributes.append(eval('[' + domain.group(2) + ']'))
                elif txt[0] == "@inputs":
                    for i in range(len(txt) - 1):
                        inputs.append(txt[i + 1].replace(',', ''))
                elif txt[0] == "@outputs":
                    outputs.append(txt[1])
            else:
                row = eval('[' + line + ']')
                if len(row) != 0:
                    X.append(row[:-1])
                    y.append(row[-1])
            line = f.readline()
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)
    return X, y, attributes, inputs, outputs


def normalize(X, y, attributes):
    """
    Normalizes the dataset in order to be handled by the algorithm

    :param X: input data
    :param y: output data
    :param attributes: range of values for each feature
    :return:
        X and y normalized such that 0 <= X[i,j] <= 1 for every i,j
    """
    n_features = X.shape[1]
    min_values = np.zeros([n_features])
    max_values = np.zeros([n_features])
    for j in range(n_features):
        min_values[j] = attributes[j][0]
        max_values[j] = attributes[j][1]
        for i in range(len(X[:, j])):
            X[i, j] = (X[i, j] - min_values[j]) / (max_values[j] - min_values[j])
    return X, y


class MPAES_RCS(MOEL_FRBC):

    def __init__(self, M=100, Amin=1, capacity=64, divisions=8, variator=RCSVariator(),
                 initializer=RCSInitializer(), objectives=('accuracy', 'trl'),
                 moea_type='mpaes22'):
        """

        :param M: Number of maximum rules that a solution (Fuzzy classifier) can have
        :param Amin: minimum number of antecedents for each rule
        :param capacity: maximum capacity of the archive
        :param divisions: number of divisions for the grid within the archive
        :param variator: genetic operator (embeds crossover and mutation functionalities)
        :param initializer: define the initial state of the archive
        :param objectives: list of objectives to minimize/maximize. As for now, only the pairs:
            ('accuracy', 'trl') and ('auc', 'trl') are provided
        :param moea_type: the Multi-Objective Evolutionary Algorithm from the ones available. If not
            provided, the default one is MPAES(2+2)
        """
        super(MPAES_RCS, self).__init__([])
        self.M = M
        self.Amin = Amin
        self.capacity = capacity
        self.divisions = divisions

        self.objectives = objectives
        self.variator = variator
        self.initializer = initializer
        self.moea_type = moea_type

    def _initialize(self, X, y):
        self.initializer.fit_tree(X, y)
        J = self.initializer.get_rules()
        splits = self.initializer.get_splits()
        # Find initial set of rules and fuzzy partitions
        # MOEFS problem
        self.problem = RCSProblem(self.Amin, self.M, J, splits, self.objectives)
        self.problem.set_training_set(X, y)

    def _choose_algorithm(self):
        if self.moea_type == 'nsga2':
            self.algorithm = NSGAIIS(self.problem, population_size=self.capacity,
                                     generator=MOEAGenerator(), selector=RandomSelector(),
                                     variator=self.variator,
                                     archive=AdaptiveGridArchive(self.capacity, 2, self.divisions))
        elif self.moea_type == 'nsga3':
            self.algorithm = NSGAIIIS(self.problem, 12, 0, generator=MOEAGenerator(),
                                      selector=RandomSelector(), variator=self.variator)
        elif self.moea_type == 'gde3':
            self.algorithm = GDE3S(self.problem, self.capacity, generator=MOEAGenerator(), variator=self.variator)
        elif self.moea_type == 'ibea':
            self.algorithm = IBEAS(self.problem, population_size=self.capacity,
                                   generator=MOEAGenerator(), variator=self.variator)
        elif self.moea_type == 'moead':
            self.algorithm = MOEADS(self.problem, generator=MOEAGenerator(), variator=self.variator)
        elif self.moea_type == 'spea2':
            self.algorithm = SPEA2S(self.problem, self.capacity, generator=MOEAGenerator(), variator=self.variator)
        elif self.moea_type == 'epsmoea':
            self.algorithm = EpsMOEAS(self.problem, [0.05], self.capacity, generator=MOEAGenerator(),
                                      selector=RandomSelector(), variator=self.variator)
        else:
            self.algorithm = MPAES2_2(self.problem, self.variator, self.capacity, self.divisions)

    def fit(self, X, y, max_evals=10000):

        self._initialize(X, y)

        self._choose_algorithm()
        self.algorithm.run(condition=max_evals)

        solutions = []
        self.archive = []
        if hasattr(self.algorithm, 'archive'):
            solutions = copy.deepcopy(self.algorithm.archive)
        elif hasattr(self.algorithm, 'population'):
            solutions = copy.deepcopy(self.algorithm.population)
        for solution in solutions:
            self.archive.append(solution)
        self._sort_archive()
        self.classifiers = [self.problem.decode(solution) for solution in self.archive]
        return self

    def _sort_archive(self):
        archive_size = len(self.archive)
        for i in range(archive_size - 1):
            max_index = i
            for j in range(i + 1, archive_size):
                acc_j = 1.0 - self.archive[j].objectives[0]
                acc_max = 1.0 - self.archive[max_index].objectives[0]
                if acc_j > acc_max:
                    max_index = j
            if max_index != i:
                self.archive[max_index], self.archive[i] = self.archive[i], self.archive[max_index]

    def cross_val_score(self, X, y, num_fold=5, seed=0, filename=''):
        n_sols = 3
        n_stats = 8
        np.random.seed(seed)
        random.seed(seed)
        if X.shape[0] != y.shape[0]:
            print("X and Y must have the same amount of samples.")
            return
        kf = KFold(n_splits=num_fold, shuffle=True)
        scores = np.zeros([n_stats, n_sols, num_fold])
        scores_archives = np.zeros([len(milestones), n_stats, n_sols, num_fold])

        for i, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            start = time.time()
            if not is_object_present(filename + '_fold' + str(i) + '_s' + str(seed)):
                my_fold = self.fit(X_train, y_train, 50000)
                store_object(self, filename + '_fold' + str(i) + '_s' + str(seed))
            else:
                my_fold = load_object(filename + '_fold' + str(i) + '_s' + str(seed))
            end = time.time()

            classifiers = my_fold.classifiers

            indexes = [0, int(len(classifiers) / 2), -1]
            snapshots = my_fold.algorithm.snapshots
            for j in range(len(snapshots)):
                print(milestones[j])
                snapshot = snapshots[j]
                archive_ind = [0, int(len(snapshot) / 2), -1]
                res = sorted(range(len(snapshot)), key=lambda q: snapshot[q].objectives[0])
                for k in range(n_sols):
                    classifier = my_fold.problem.decode(snapshot[res[archive_ind[k]]])
                    y_train_pred = classifier.predict(X_train)
                    y_test_pred = classifier.predict(X_test)
                    scores_archives[j, 0, k, i] = (sum(y_train_pred == y_train) / len(y_train)) * 100.0
                    scores_archives[j, 0, k, i] = (sum(y_test_pred == y_test) / len(y_test)) * 100.0
                    scores_archives[j, 0, k, i] = classifier.trl()
                    scores_archives[j, 0, k, i] = classifier.num_rules()
                    scores_archives[j, 0, k, i] = len(classifiers)
                    prec, recall, fscore, _ = metrics.precision_recall_fscore_support(y_test_pred, y_test,
                                                                                      average='weighted')
                    scores_archives[j, 0, k, i] = prec
                    scores_archives[j, 0, k, i] = recall
                    scores_archives[j, 0, k, i] = fscore
            for k in range(n_sols):
                classifier = classifiers[indexes[k]]
                y_train_pred = classifier.predict(X_train)
                y_test_pred = classifier.predict(X_test)
                scores[0, k, i] = (sum(y_train_pred == y_train) / len(y_train)) * 100.0
                scores[1, k, i] = (sum(y_test_pred == y_test) / len(y_test)) * 100.0
                scores[2, k, i] = classifier.trl()
                scores[3, k, i] = classifier.num_rules()
                scores[4, k, i] = len(classifiers)
                prec, recall, fscore, _ = metrics.precision_recall_fscore_support(y_test_pred, y_test,
                                                                                  average='weighted')
                scores[5, k, i] = prec
                scores[6, k, i] = recall
                scores[7, k, i] = fscore

        return scores, scores_archives

    def _from_position_to_index(self, position):
        if len(self.archive) > 0:
            index = {'first': 0, 'median': int(len(self.archive) / 2), 'last': -1}
            return index[position.lower()]
        return None

    def show_model(self, position='first', inputs=None, outputs=None, f=None):
        index = self._from_position_to_index(position)
        if index is not None:
            self.classifiers[index].show_RB(inputs, outputs, f)
            self.classifiers[index].show_DB(inputs)

    def _plot_archive(self, archive, x=None, y=None, label='', marker='o'):
        archive_size = len(archive)
        problem = self.problem
        objectives = problem.objectives
        values = np.zeros([archive_size, len(objectives), 2])
        for i in range(archive_size):
            classifier = problem.decode(archive[i])
            for j in range(len(objectives)):
                values[i, j, 0] = archive[i].objectives[j]
                if x is not None and y is not None:
                    values[i, j, 1] = problem.evaluate_obj(classifier, objectives[j], x, y)
                if objectives[j] == 'auc' or objectives[j] == 'accuracy':
                    values[i, j, 0] = (1.0 - values[i, j, 0]) * 100.0
                    if x is not None and y is not None:
                        values[i, j, 1] = (1.0 - values[i, j, 1]) * 100.0

        trl_index = objectives.index('trl')
        other_obj = np.setdiff1d(np.arange(len(objectives)), trl_index)[0]
        coords = np.array(
            sorted(zip(values[:, trl_index, 0], values[:, other_obj, 0]), key=lambda t: (t[0], 100 - t[1])))
        _, indices = np.unique(coords[:, 0], return_index=True)

        plt.plot(coords[indices, 0], coords[indices, 1], marker=marker, linestyle='dotted', markersize=11,
                 label='train')
        if x is not None and y is not None:
            coords = np.array(
                sorted(zip(values[:, trl_index, 1], values[:, other_obj, 1]), key=lambda t: (t[0], 100 - t[1])))
            _, indices = np.unique(coords[:, 0], return_index=True)
            plt.plot(coords[indices, 0], coords[indices, 1], marker=marker, linestyle='dotted', markersize=11,
                     label='test')

    def show_pareto(self, x=None, y=None, path=''):
        set_default_plot_style()

        plt.figure()
        objectives = self.algorithm.problem.objectives
        if len(self.archive) > 0:
            self._plot_archive(self.archive, x, y)
        # Add a legend
        plt.legend(loc='best')
        plt.xlabel(objectives[1])
        plt.ylabel(objectives[0])
        plt.xlim(left=0)
        # Show the plot
        plt.show()
        if path != '':
            plt.savefig(path + '_pareto.svg')

    def show_pareto_archives(self, x, y, path=''):
        set_default_plot_style()

        fig = plt.figure()
        objectives = self.problem.objectives
        for i in range(len(self.algorithm.snapshots)):
            marker = '+' if i % 2 else 'x'
            self._plot_archive(self.algorithm.snapshots[i], x, y, label=str(milestones[i]), marker=marker)
        self._plot_archive(self.archive, x, y, str(self.algorithm.nfe), '*')
        # Add a legend
        plt.legend(loc='best')
        plt.xlabel(objectives[1])
        plt.ylabel(objectives[0])
        plt.xlim(left=0)
        # Show the plot
        plt.show()
        if path != '':
            plt.savefig(path + '_archives.svg')

    def __getitem__(self, position):
        if isinstance(position, str):
            index = self._from_position_to_index(position)
        elif isinstance(position, int):
            index = position
        else:
            return None
        return self.classifiers[index]
