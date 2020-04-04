from __future__ import print_function
from skmoefs.discretization import discretizer_fuzzy as df
from skmoefs.fuzzysets.TriangularFuzzySet import TriangularFuzzySet
from skmoefs.fuzzysets.SingletonFuzzySet import SingletonFuzzySet
from skmoefs.fuzzysets.UniverseFuzzySet import UniverseFuzzySet
from skmoefs.fuzzysets.TrapezoidalFuzzySet import TrapezoidalFuzzySet
import numpy as np


class FuzzyImpurity(object):
    """Fuzzy impurity class.
    """

    @staticmethod
    def calculate(counts, totalCount):
        """Evaluate fuzzy impurity, given a list of fuzzy cardinalities.

        Parameters
        ----------
        counts : array-like
            An array, containing the fuzzy cardinality for a fuzzy set for
            each class.
        totalCount : float
            Global cardinality for a fuzzy set.

        Returns
        -------
        fuzzy_impurity : float
            Fuzzy impurity evaluated on the given set.
        """
        if totalCount == 0:
            return 0.0
        numClasses = len(counts)
        impurity = 0.0
        classIndex = 0

        while (classIndex < numClasses):
            classCount = counts[classIndex]
            if classCount != 0:
                freq = classCount / float(totalCount)
                impurity -= freq * np.log2(freq)
            classIndex += 1
        return impurity
    
    
def findContinous(X):
    """
    Parameters
    ----------
    X, np.array, shape (Nsample, Nfeatures)

    Returns
    -------
    list, len Nfeatures
        A list containing True if the corresponding element is regarded as continous,
        False otherwise
    """
    N, M = X.shape
    red_axis = np.array([len(np.unique(X[:, k])) for k in range(M)]) / float(N)
    return list(red_axis > 0.05)


class decisionNode:

    def __init__(self, feature=-1, isLeaf=False, results=None,
                 child=None, weight=None ,fSet=None):
        """Decision Node.
        """
        # Feature corresponding to the set on the node.
        # it is not the feature used for splitting the child.
        self.feature = feature
        
        self.fSet = fSet
        self.isLeaf = isLeaf
        self.results = results
        self.child = child
        self.weight = weight

    def predict(self, observation, currMD, numClasses):
        if currMD > 0:
            if self.isLeaf:
                return self.results * FMDT.tNorm(currMD, self.fSet.membershipDegree(observation[self.feature]))

            else:
                v = np.zeros(numClasses)
                childSum = map(lambda child: child.predict(observation, FMDT.tNorm(currMD,
                                                           self.fSet.membershipDegree(observation[self.feature])),
                                                           numClasses), self.child)
                for k in childSum:
                    v += k
                return v
        else:
            return np.zeros(numClasses)

    def _printSubTree(self, indentFactor=0):
        prefix = "|    " * indentFactor
        stringa = ""
        if self.isLeaf:
            stringa += " : " + str(self.results)
        else:
            if self.child is None:
                stringa += ""
            else:
                for k in range(len(self.child)):
                    stringa += "\n" + prefix + "Feature: " + str(self.child[k].feature) + " - [ " + str(
                        self.child[k].fSet.__str__())+ ' ]'
                    stringa += self.child[k]._printSubTree(indentFactor + 1)
        return stringa

    def _ruleMine(self, prec="", ruleArray=None):
        if ruleArray == None:
            ruleArray = []
        if self.isLeaf:
            if sum(self.results) > 0:
                prec += " then " + ("[" + ', '.join(['%.2f'] * len(self.results)) + "]") % tuple(self.results)
                ruleArray.append(prec.replace('\n', ''))
            else:
                pass
        else:
            if self.child is None:
                pass
            else:
                if prec != "":
                    prec += " and "
                else:
                    prec += "if "
                for k in range(len(self.child)):
                    precL = prec + "A" + str(self.child[k].feature) + " is " + "FS" + str(self.child[k].fSet.index)
                    self.child[k]._ruleMine(precL, ruleArray)
        return ruleArray

    def _csv_ruleMine(self,numFeat, prec=[], ruleArray=None):
        if ruleArray == None:
            ruleArray = []
        if self.isLeaf:
            if sum(self.results) > 0:
                prec[-1] = np.argmax(self.results) + 1
                ruleArray.append(prec)
            else:
                pass
        else:
            if self.child is None:
                pass
            else:
                if prec != []:
                    pass
                else:
                    prec = [0] * (numFeat + 1)
                for k in range(len(self.child)):
                    precL = prec[:]
                    precL[self.child[k].feature] = self.child[k].fSet.index + 1
                    self.child[k]._csv_ruleMine(numFeat, precL[:], ruleArray)
    
        return ruleArray

    def _num_leaves(self):
        if self.isLeaf == 1 and np.sum(self.results) != 0:
            leaf = 1.
        else:
            leaf = 0.
        if not self.child is None:
            childSum = map(lambda child: child._num_leaves(), self.child)
            for k in childSum:
                leaf += k
        return leaf

    def _sons(self):
        array = []
        array = self.child
        return array

    def _ruleLength(self, depth):
        if self.isLeaf:
            if (sum(self.results) <= 0):
                return 0.
            return depth
        else:
            return depth + sum([child._ruleLength(depth+1) for child in self.child])
            
        
    def _numDescendants(self, empty):
        if self.isLeaf:
            if (sum(self.results) <= 0) and (not empty):
                return 0
            return 1.
        else:
            return 1 + sum([child._numDescendants(empty) for child in self.child])


class FMDT(object):
    """Class implementing a multi-split fuzzy decision tree.
    """

    def __init__(self, max_depth=5, discr_minImpurity=0.02,
                 discr_minGain=0.01, minGain=0.01,
                 minNumExamples=2, max_prop=1.0, priorDiscretization=False,
                 discr_threshold=0, verbose=False, features='all'):
        """FMDT with fuzzy discretization on the node.

        Parameters
        ----------
        max_depth : int, optional, default = 5
            maximum tree depth.
        discr_minImpurity : int, optional default = 0.02
            minimum imputiry for a fuzzy set during discretization
        discr_minGain : float, optional, default = 0.01
            minimum entropy gain during discretization
        discr_threshold : int, optional, default = 0
            if discr_threshold != 0  the discretization is stopped
            at n = discr_threshold + 2 fuzzy sets.
        minGain : float, optional, default = 0.01
            minimum entropy gain for a split during tree induction
        minNumExample : int, optional, default = 2
            minimum number of example for a node during tree induction
        max_prop : float, optional, default = 1.0
            min proportion of samples pertaining to the same class in the same node
            for stop splitting.
        prior_discretization : boolean, optional, default = True
            define whether performing prior or node level discretization

        Notes
        -----
        When using the  FMDT in a fuzzy random forest implementation, programmers
        are encouraged to use priorDiscretization = True and providing the cPoints
        upon fit. This would help in speeding up computation.
        """

        self.max_depth = max_depth
        self.verbose = verbose
        self.discr_minImpurity = discr_minImpurity
        self.discr_minGain = discr_minGain
        self.minGain = minGain
        self.minNumExamples = minNumExamples
        self.priorDiscretization = priorDiscretization
        self.max_prop = max_prop
        self.discr_threshold = discr_threshold
        self.features = features

    def fit(self, X, y, continous=None, numClasses=None, cPoints=None, ftype='triangular',trpzPrm=-1):
        """Build a multi-way fuzzy decision tree classifier from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (real numbers). Use ``dtype=np.float64`` and
            ``order='C'`` for maximum efficiency.

        numClasses : int
            Number of classes for the dataset.

         cPoints: array-like, shape = [n_features], optional
            Array of array of cut points.

        Returns
        -------
        self : object
            Returns self.
        """
        assert ftype in ['triangular', 'trapezoidal'], "Invalid fuzzy set type: %s" %ftype
        if not numClasses is None:
            self.K = numClasses
        else:
            self.K = int(np.max(y) + 1)

        if continous is None:
            self.cont = np.array(findContinous(X))
        else:
            self.cont = np.array(continous)

        self.N, self.M = X.shape

        if self.priorDiscretization:

            if not cPoints is None:
                self.cPoints = cPoints

            # Executing discretization
            else:
                discr = df.FuzzyMDLFilter(self.K, X, y, self.cont,
                                          minGain=self.discr_minGain,
                                          minImpurity=self.discr_minImpurity,
                                          threshold=self.discr_threshold)
                self.cPoints = np.array(discr.run())
        self.fSets = []
        
        for k, points in enumerate(self.cPoints):
            if not continous[k] == True:
                if self.cPoints[k]:
                    points = self.cPoints[k]
                else:
                    points = np.unique(X[:,k])
                    self.cPoints[k] = points
                self.fSets.append(SingletonFuzzySet.createFuzzySets(points))
                
            elif len(points) == 0:
                self.fSets.append([])
            else:
                if ftype == 'triangular':
                    self.fSets.append(TriangularFuzzySet.createFuzzySets(points, isStrongPartition=True))
                elif ftype == 'trapezoidal':
                    self.fSets.append(TrapezoidalFuzzySet.createFuzzySets(points, isStrongPartition=True, trpzPrm=trpzPrm))
                    pass
                
        tree = self.buildtree(np.concatenate((X, y.reshape(y.shape[0], 1)), axis=1), depth=0,
                              fSet=UniverseFuzzySet.createFuzzySet())
        self.tree = tree
        return self

    def buildtree(self, rows, scoref=FuzzyImpurity, depth=0,
                  memb_degree=None, leftAttributes=None, feature=-1, fSet = None):

        # Attributes are "consumed" on a given path from root to leaf,
        # Please note: [] != None
        if leftAttributes == None:
            leftAttributes = list(range(self.M))

        # Membership degree of the given set of samples for the current node
        if memb_degree is None:
            memb_degree = np.ones(len(rows))

        # Set up some variables to track the best criteria
        best_gain = 0.0
        best_criteria = None
        best_sets = None

        # Class counts on the current node
        class_counts = self.classCounts(rows, memb_degree)
        if depth == 0:
            self.dom_class = np.argmax(class_counts)
        # Stop splitting if
        # - the proportion of dataset of a class k is greater than a threshold max_prop
        # - the cardinality of the dataset in the node is lower then a threshold self.min_num_examples
        # - there are no attributes left
        with np.errstate(divide='ignore', invalid='ignore'):
            if (np.any(class_counts / float(np.sum(class_counts)) >= self.max_prop) or
                        np.sum(class_counts) < self.minNumExamples or leftAttributes == []):
                return decisionNode(results=np.nan_to_num(class_counts / float(np.sum(class_counts))), feature=feature,
                                    isLeaf=True, weight=memb_degree.sum(), fSet = fSet)

        # Calculate entropy for the node
        current_score = scoref.calculate(class_counts, sum(class_counts))

        # Iterate among features
        gain = best_gain
        for col in leftAttributes:
            if list(self.fSets[col]):
                row_vect, memb_vect = self.__multidivide(rows, memb_degree, col, self.fSets[col])
                classes = np.array([np.sum(k) for k in memb_vect])
                pj = classes / np.sum(classes)
                cCounts = [self.classCounts(r, m) for r, m in zip(row_vect, memb_vect)]
                I = np.array([scoref.calculate(c, sum(c)) for c in cCounts])
                gain = current_score - np.sum(pj * I)

            if gain > best_gain:
                best_gain = gain
                best_criteria = col
                best_sets = (memb_vect, row_vect)

        if best_gain > self.minGain and depth < self.max_depth:
            memb_val, row_vect = best_sets
            child_list = []
            leftAttributes.pop(leftAttributes.index(col))
            for k in range(len(memb_val)):
                child_list.append(
                    self.buildtree(rows=row_vect[k], memb_degree=memb_val[k], leftAttributes=leftAttributes[:],
                                   depth=depth + 1, feature=best_criteria,fSet = self.fSets[best_criteria][k]))
            return decisionNode(results=np.nan_to_num(class_counts /
                float(np.sum(class_counts))), feature=feature, child=child_list, weight=memb_degree.sum(),
                                fSet = fSet)

        else:
            return decisionNode(results=np.nan_to_num(class_counts / float(np.sum(class_counts))), feature=feature,
                                isLeaf=True, weight=memb_degree.sum(), fSet = fSet)

    def __multidivide(self, rows, membership, column, fSets):
        
        memb_vect = []
        row_vect = []
        for fSet in fSets:
            mask = list(map(lambda x: fSet.isInSupport(x), rows[:,column]))
            row_vect.append(rows[mask,:])
            memb_vect.append(FMDT.tNorm(list(map(lambda x: fSet.membershipDegree(x), rows[mask,column])), membership[mask]))
            
        return row_vect, memb_vect
        
    
    def classCounts(self, rows, memb_degree):
        """Returns the sum of the membership degree for each class in a given set.
        """
        labels = rows[:, -1]
        numClasses = self.K
        llab = list(set(labels))
        llab.sort()
        counts = np.zeros(numClasses, dtype=float)
        for k in range(len(llab)):
            ind = int(llab[k])
            counts[ind] = np.nan_to_num(np.sum(memb_degree[labels == llab[k]]))
            # counts[ind] = np.nan_to_num(np.sum((memb_degree!=0)[labels == llab[k]]))
        return counts

    def predict(self, X):
        """Predict class or regression value for X.

        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.

        Parameters
        ----------
        X : array-like  of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes, or the predict values.
        """
        return np.array(list(map(lambda X: self.classify(X, self.tree, self.K), X)))

    def predictRF(self, obs):
        """Prediction for RF.
        """
        prediction = self.tree.predict(obs, 1., self.K)
        return prediction

    def printTree(self):
        """Print the decision tree.
        """

        stringTree = "Fuzzy Decision tree \n" + self.tree._printSubTree(0)

        print(stringTree)

    def numLeaves(self):
        """ Number of non-empty leaves in the three.
        """
        return self.tree._num_leaves()

    def numNodes(self, empty=False):
        """Number of nodes in the three.
        """

        return np.sum(list(map(lambda x: x._numDescendants(empty), self.tree._sons())))
    
    def totalRuleLength(self):
        """Rule length.
        """
        return self.tree._ruleLength(depth=0)

    def classify(self, observation, tree, numClass):
        prediction = tree.predict(observation, 1., numClass)
        if np.sum(prediction) > 0:
            return np.argmax(prediction)
        else:
            return -1
    @staticmethod
    def tNorm(array1, array2, tnorm='product'):
        """Method for calculating various elemntwise t_norms.

        Parameters
        ----------
        array1, numpy.array()
            First array
        array2, numpy.array()
            Second array
        """
        if tnorm == 'product':
            return array1 * array2
        if tnorm == 'min':
            return np.minimum(array1, array2)
