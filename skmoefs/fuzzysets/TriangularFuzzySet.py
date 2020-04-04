from __future__ import division
from skmoefs.fuzzysets.FuzzySet import FuzzySet

class TriangularFuzzySet(FuzzySet):

    def __init__(self, a, b, c, index=None):
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.__leftSupportWidth = b - a
        self.__rightSupportWidth = c - b
        self.left = self.a
        self.right = self.b
        if index is not None:
            self.index = index

    def __str__(self):
        return "a=%f, b=%f, c=%f"%(self.a, self.b, self.c)

    def isInSupport(self, xi):
        return xi > self.a and xi < self.c

    def membershipDegree(self, xi):
        if self.isInSupport(xi):
            if (xi <= self.b and self.a == -float('inf')) or (xi >= self.b and self.c == float('inf')):
                return 1.0
            else:
                if ( xi <= self.b):
                    uAi = (xi - self.a) / self.__leftSupportWidth
                else:
                    uAi = 1.0 - ((xi - self.b) / self.__rightSupportWidth)
                return uAi
        else:
            return 0.0

    def isFirstOfPartition(self):
        return self.a == -float('inf')

    def isLastOfPartition(self):
        return self.c == float('inf')

    @staticmethod
    def createFuzzySet(params):
        assert len(params) == 3,  "Triangular Fuzzy Set Builder requires three parameters (left, peak and rigth)," \
                                  " but %d values have been provided." % len(params)
        sortedParameters = sorted(params)
        return TriangularFuzzySet(sortedParameters[0], sortedParameters[1], sortedParameters[2])

    @staticmethod
    def createFuzzySets(params, isStrongPartition = False):
        assert len(params) > 1,  "Triangular Fuzzy Set Builder requires at least two points," \
                                  " but %d values have been provided." % len(params)
        sortedPoints = sorted(params)
        if isStrongPartition:
            return TriangularFuzzySet.createFuzzySetsFromStrongPartition(sortedPoints)
        else:
            return TriangularFuzzySet.createFuzzySetsFromNoStrongPartition(sortedPoints)

    @staticmethod
    def createFuzzySetsFromStrongPartition(points):
        fuzzySets = []
        fuzzySets.append(TriangularFuzzySet(-float('inf'), points[0], points[1],index=0))

        for index in range(1, len(points)-1):
            fuzzySets.append(TriangularFuzzySet(points[index - 1], points[index], points[index + 1], index))

        fuzzySets.append(TriangularFuzzySet(points[-2], points[-1], float('inf'), len(points)-1))
        return fuzzySets

    @staticmethod
    def createFuzzySetsFromNoStrongPartition(points):
        assert (len(points)-4)%3 == 0, "Triangular Fuzzy Set Builder requires a multiple of three plus 4 " \
                                       "as valid number of points, but %d points have been provided."% len(points)
        fuzzySets = []
        fuzzySets.append(TriangularFuzzySet(-float('inf'), points[0], points[1]))

        for index in range(1,len(points)-1):
            indexPoints = index*3
            fuzzySets.append(TriangularFuzzySet(points[indexPoints - 1], points[indexPoints], points[indexPoints + 1]))

        fuzzySets.append(TriangularFuzzySet(points[-2], points[-1], float('inf')))
        return fuzzySets

