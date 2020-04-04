from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score

class ClassificationRule():

    def __init__(self, antecedent, fuzzyset, consequent, weight=1.0, label=None):
        self.antecedent = antecedent
        self.fuzzyset = fuzzyset
        self.consequent = consequent
        self.weight = weight

        if label is not None:
            self.label = label
        else:
            self.label = id(self)

@jit(nopython=True)
def membership_value(mf, value):
    if len(mf) == 3:
        if mf[0] == mf[1]: # left triangular
            if value < mf[0]:
                return 1.0
            elif value > mf[2]:
                return 0.0
            else:
                return 1.0 - ((value - mf[1]) / (mf[2] - mf[1]))
        elif mf[1] == mf[2]: # right triangular
            if value < mf[0]:
                return 0.0
            elif value > mf[2]:
                return 1.0
            else:
                return (value - mf[0]) / (mf[1] - mf[0])
        else: # triangular
            if value < mf[0] or value > mf[2]:
                return 0.0
            elif value <= mf[1]:
                return (value - mf[0]) / (mf[1] - mf[0])
            elif value <= mf[2]:
                return 1.0 - ((value - mf[1]) / (mf[2] - mf[1]))
    # Not implemented
    return 0.0

@jit(nopython=True)
def predict_fast(x, ant_matrix, cons_vect, weights, part_matrix):
    """

    :param x: input matrix NxM where N is the number of samples and M is the number of features
    :param ant_matrix: antecedents of every rule in the RB
    :param cons_vect: consequents of every rule in the RB
    :param weights:
    :param part_matrix: partitions of fuzzysets
    :return:
    """
    y = np.zeros(x.shape[0])
    for i in range(y.shape[0]):
        best_match_index = 0
        best_match = 0
        for j in range(cons_vect.shape[0]):
            matching_degree = 1.0
            for k in range(ant_matrix.shape[1]):
                if not np.isnan(ant_matrix[j][k]):
                    ant = part_matrix[k][ant_matrix[j][k]:ant_matrix[j][k]+3]
                    m_degree = membership_value(ant, x[i][k])
                    matching_degree *= m_degree
            if weights[j] * matching_degree > best_match:
                best_match_index = j
        y[i] = cons_vect[best_match_index]
    return y

@jit(nopython=True)
def compute_weights_fast(train_x, train_y, ant_matrix, cons_vect, part_matrix):
    """

    :param train_x: Training input
    :param train_y: Training output
    :param ant_matrix: antecedents
    :param cons_vect: consequents
    :param part_matrix: partitions of fuzzysets
    :return: for each rule in the RB, compute the weight from the provided
            training set
    """
    weights = np.ones(ant_matrix.shape[0])
    for i in range(ant_matrix.shape[0]):
        matching = 0.0
        total = 0.0
        for j in range(train_y.shape[0]):
            matching_degree = 1.0
            for k in range(ant_matrix.shape[1]):
                if not np.isnan(ant_matrix[j][k]):
                    ant = part_matrix[k][ant_matrix[i][k]:ant_matrix[i][k] + 3]
                    m_degree = membership_value(ant, train_x[j][k])
                    matching_degree *= m_degree
            if train_y[j] == cons_vect[i]:
                matching += matching_degree
            total += matching_degree
        weights[i] = total if total == 0 else matching / total
    return weights

class FuzzyRuleBasedClassifier():
    """
    Fuzzy Rule-Based Classifier class
    """

    def __init__(self, rules, partitions):

        """

        :param rules: a list of ClassificationRule objects
        :param partitions: fuzzyset partitions for each fuzzy input
        """

        # RB info
        self.rules = rules
        # DB info
        self.partitions = partitions

        # RB and DB information are converted into NumPy matrices
        self.ant_matrix = np.empty((len(rules), len(partitions)))
        self.ant_matrix[:] = np.NaN
        self.cons_vect = np.empty((len(rules)))
        self.weights = np.ones((len(rules)))
        self.part_matrix = np.asmatrix(partitions)
        self.part_matrix = np.hstack((self.part_matrix[:, 0], self.part_matrix[:], self.part_matrix[:, -1]))
        for i, rule in enumerate(self.rules):
            for key in rule.antecedent:
                self.ant_matrix[i, key] = rule.fuzzyset[key] - 1
            self.cons_vect[i] = rule.consequent
            self.weights[i] = rule.weight

    def addrule(self, new_rule):
        self.rules.append(new_rule)

    def num_rules(self):
        return len(self.rules)

    def predict(self, x):
        return predict_fast(x, self.ant_matrix, self.cons_vect, self.weights, self.part_matrix)

    def compute_weights(self, train_x, train_y):
        self.weights = compute_weights_fast(train_x, train_y, self.ant_matrix, self.cons_vect, self.part_matrix)

    def trl(self):
        n_antecedents = [len(rule.antecedent) for rule in self.rules]
        return np.sum(n_antecedents)

    def accuracy(self, x, y):
        return sum(self.predict(x) == y) / len(y)

    def auc(self, x, y):
        y_pred = self.predict(x)
        # Binarize labels (One-vs-All)
        lb = LabelBinarizer()
        lb.fit(y)
        # Transform labels
        y_bin = lb.transform(y)
        y_pred_bin = lb.transform(y_pred)
        return roc_auc_score(y_bin, y_pred_bin, average='macro')

    def _get_labels(self, size):
        if size == 3:
            return ['L', 'M', 'H']
        if size == 5:
            return ['VL', 'L', 'M', 'H', 'VH']
        if size == 7:
            return ['VL', 'L', 'ML', 'M', 'MH', 'H', 'VH']

    def show_RB(self, inputs, outputs, f=None):
        if f:
            f.write('RULE BASE\n')
            startBold = ''
            endBold = ''
        else:
            print('RULE BASE')
            startBold = '\033[1m'
            endBold = '\033[0m'
        if_keyword = startBold + 'IF' + endBold
        then_keyword = startBold + 'THEN' + endBold
        is_keyword = startBold + 'is' + endBold

        for i, rule in enumerate(self.rules):
            if_part = if_keyword + ' '
            count = 0
            for key in rule.antecedent:
                size = len(self.partitions[key])
                labels = self._get_labels(size)
                if count > 0:
                    if_part += startBold + ' AND ' + endBold
                count += 1
                if inputs is None:
                    feature = 'X_' + str(key + 1)
                else:
                    feature = inputs[key]
                if_part += feature
                if_part += ' ' + is_keyword + ' ' + labels[rule.fuzzyset[key]-1] + ' '
            if outputs is None:
                output = 'Class'
            else:
                output = outputs[0]
            then_part = then_keyword + ' ' + output + ' is ' + str(rule.consequent)
            if f:
                f.write(str(i+1) + ':\t' + if_part + then_part + '\n')
            else:
                print(str(i+1) + ':\t' + if_part + then_part)

    def show_DB(self, inputs):
        """

        :param inputs: names of the input variables
        :return: for each input, plot a graph representing the membership functions of
                fuzzyset partitions
        """

        plt.rc('font', size=10)  # controls default text sizes
        plt.rc('axes', titlesize=14)  # fontsize of the axes title
        plt.rc('axes', labelsize=16)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=14)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=14)  # fontsize of the tick labels
        plt.rc('legend', fontsize=14)  # legend fontsize
        plt.rc('figure', titlesize=14)  # fontsize of the figure title
        for k, partition in enumerate(self.partitions):
            plt.figure()
            anchors = np.concatenate(([partition[0]], partition, [partition[-1]]))
            fuzzyset_size= len(anchors) - 2
            triangle = np.array([0.0, 1.0, 0.0])
            for i in range(fuzzyset_size):
                plt.plot(anchors[i:i+3], triangle, linestyle='solid', linewidth=2, color='k')

            # Add a legend
            if inputs is not None:
                xLabel = inputs[k]
            else:
                xLabel = 'X_' + str(k+1)
            plt.xlabel(xLabel)
            plt.ylim([0.0, 1.1])
            plt.xlim([0.0, 1.0])

            # Show the plot
            plt.show()
            plt.close()