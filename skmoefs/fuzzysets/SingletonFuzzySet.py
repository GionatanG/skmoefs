from __future__ import division
from skmoefs.fuzzysets.FuzzySet import FuzzySet
import numpy as np


class SingletonFuzzySet(FuzzySet):

    def __init__(self, value,index=None):
        self.value = value
        self.left = self.value
        self.right = self.value
        if index is not None:
            self.index = index

    def __str__(self):
        return "value=%f"%self.value

    def isInSupport(self, xi):
        return xi == self.value

    def isFirstOfPartition(self):
        return self.value == -float('inf')

    def isLastOfPartition(self):
        return self.value == float('inf')

    def membershipDegree(self, xi):
        if self.isInSupport(xi):
            return 1.0
        else:
            return 0.0

    @staticmethod
    def createFuzzySet(params,index=None):
        assert np.isscalar(params),  "Invalid parameter for Singleton Fuzzy Set!"
        return SingletonFuzzySet(params,index)

    @staticmethod
    def createFuzzySets(points, isStrongPartition = False):
        return list(map(lambda point: SingletonFuzzySet.createFuzzySet(point[1],point[0]), enumerate(points)))

