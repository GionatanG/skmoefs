from __future__ import division
from skmoefs.fuzzysets.FuzzySet import FuzzySet

class UniverseFuzzySet(FuzzySet):

    def __init__(self):
        pass

    def __str__(self):
        return "Universe Fuzzy Set"

    def isInSupport(self, xi):
        return 1.0

    def isFirstOfPartition(self):
        return True

    def isLastOfPartition(self):
        return True

    def membershipDegree(self, xi):
        return 1.0
    @staticmethod
    def createFuzzySet():
        return UniverseFuzzySet()

