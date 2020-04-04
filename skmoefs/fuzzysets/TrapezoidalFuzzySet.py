from __future__ import division
from skmoefs.fuzzysets.FuzzySet import FuzzySet

class __TrapezoidalFuzzySet(FuzzySet):

    def __init__(self, a, b, c, d, index=None):
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)
        self.__leftSupportWidth = b - a
        self.__rightSupportWidth = d - c
        self.left = self.a
        self.right = self.b
        if index is not None:
            self.index = index

    def __str__(self):
        return "a=%f, b=%f, c=%f, d=%f"%(self.a, self.b, self.c, self.d)

    def isInSupport(self, xi):
        return xi > self.a and xi < self.d

    def membershipDegree(self, xi):
        if self.isInSupport(xi):
            if (xi <= self.b and self.a == -float('inf')) or (xi >= self.b and self.c == float('inf')):
                return 1.0
            else:
                if ( xi <= self.b):
                    uAi = (xi - self.a) / self.__leftSupportWidth
                elif ( xi >= self.c):
                    uAi = 1.0 - ((xi - self.c) / self.__rightSupportWidth)
                else:
                    uAi = 1.0
                return uAi

        else:
            return 0.0

    def isFirstofPartition(self):
        return self.a == -float('inf')

    def isLastOfPartition(self):
        return self.c == float('inf')

    @staticmethod
    def createFuzzySet(params):
        assert len(params) == 4,  "Trapezoidal Fuzzy Set Builder requires four parameters (left, peak_b, " \
                                  "peak_c and right)," \
                                  " but %d values have been provided." % len(params)
        sortedParameters = sorted(params)
        return TrapezoidalFuzzySet(sortedParameters[0], sortedParameters[1], sortedParameters[2], sortedParameters[4])

    @staticmethod
    def createFuzzySets(params, isStrongPartition = False):
        assert len(params) > 3,  "Triangular Fuzzy Set Builder requires at least four points," \
                                  " but %d values have been provided." % len(params)
        sortedPoints = sorted(params)
        if isStrongPartition:
            return TrapezoidalFuzzySet.createFuzzySetsFromStrongPartition(sortedPoints)
        else:
            return TrapezoidalFuzzySet.createFuzzySetsFromNoStrongPartition(sortedPoints)

    @staticmethod
    def createFuzzySetsFromStrongPartition(points):
        fuzzySets = []
        fuzzySets.append(TrapezoidalFuzzySet(-float('inf'), points[0], points[1], points[2], index=0))
        fCount = 1
        for index in range(2, len(points) - 2, 2):
            fuzzySets.append(TrapezoidalFuzzySet(points[index - 1], points[index],
                                                 points[index + 1], points[index+2], index//2))
            fCount += 1
        fuzzySets.append(TrapezoidalFuzzySet(points[-3], points[-2], points[-1], float('inf'), fCount))
        return fuzzySets

    @staticmethod
    def createFuzzySetsFromNoStrongPartition(points):

        assert (len(points)-6) % 4 == 0, "Triangular Fuzzy Set Builder requires a multiple of four plus 6 " \
                                       "as valid number of points, but %d points have been provided." %len(points)
        fuzzySets = []
        fuzzySets.append(TrapezoidalFuzzySet(-float('inf'), points[0], points[1], points[2]))

        for index in range(2, len(points)-2):
            indexPoints = index*4
            fuzzySets.append(TrapezoidalFuzzySet(points[indexPoints - 1], points[indexPoints], points[indexPoints + 1], points[indexPoints+2]))

        fuzzySets.append(TrapezoidalFuzzySet(points[-3], points[-2], points[-1], float('inf')))
        return fuzzySets


class TrapezoidalFuzzySet(FuzzySet):
    
    def __init__(self, a, b, c, trpzPrm, index=None):
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.trpzPrm=trpzPrm
        self.__leftPlateau =(b - a)*self.trpzPrm
        self.__rightPlateau = (c - b)*self.trpzPrm
        self.__leftSlope = (b-a)*(1-2*self.trpzPrm)
        self.__rightSlope = (c-b)*(1-2*self.trpzPrm)
        self.left = self.a
        self.right = self.b
        if index is not None:
            self.index = index

    def toDebugString(self):
        return "a=%f, b=%f, c=%f" %(self.a, self.b, self.c)

    def isInSupport(self, xi):
        if self.a == -float('inf') :
            return xi < self.c-self.__rightPlateau
        
        if self.c == float('inf'):
            return xi> self.a+self.__leftPlateau
        
        return (xi> self.a+self.__leftPlateau and xi < self.c-self.__rightPlateau)
            

    def membershipDegree(self, xi):
        
        if self.isInSupport(xi): 
            if ( xi <= self.b and self.a == -float('inf')) or (xi >= self.b and self.c == float('inf')):
                return 1.0
            
            elif (xi  <= self.b - self.__leftPlateau):
                uAi = (xi - self.a - self.__leftPlateau) / self.__leftSlope    
            
            elif (xi >= self.b-self.__leftPlateau) and (xi <= self.b + self.__rightPlateau):
                uAi=1.0   
            
            elif (xi <= self.c-self.__rightPlateau ):
                uAi = 1.0 - ((xi - self.b- self.__rightPlateau) / self.__rightSlope)
            
            else:
                uAi=0
            
            return uAi
        
        else:
            return 0.0

    def isFirstofPartition(self):
        return self.a == -float('inf')

    def isLastOfPartition(self):
        return self.c == float('inf')

    @staticmethod
    def createFuzzySet(params, trpzPrm):
        assert len(params) == 3,  " Fuzzy Set Builder requires three parameters (left, peak and rigth)," \
                                  " but %d values have been provided." % len(params)
        sortedParameters = sorted(params)
        return TrapezoidalFuzzySet(sortedParameters[0], sortedParameters[1], sortedParameters[2], trpzPrm=trpzPrm)

    @staticmethod
    def createFuzzySets(params, trpzPrm , isStrongPartition = False,):
        assert len(params) > 1,  " Fuzzy Set Builder requires at least two points," \
                                  " but %d values have been provided." % len(params)
        sortedPoints = sorted(params)
        if isStrongPartition:
            return TrapezoidalFuzzySet.createFuzzySetsFromStrongPartition(sortedPoints, trpzPrm=trpzPrm)
        else:
            return TrapezoidalFuzzySet.createFuzzySetsFromNoStrongPartition(sortedPoints, trpzPrm=trpzPrm)

    @staticmethod
    def createFuzzySetsFromStrongPartition(points, trpzPrm):
        fuzzySets = []
       
        fuzzySets.append(TrapezoidalFuzzySet(-float('inf'), points[0], points[1],index=0, trpzPrm=trpzPrm))

        for index in range(1,len(points)-1):
            fuzzySets.append(TrapezoidalFuzzySet(points[index - 1], points[index], points[index + 1], trpzPrm, index))

        fuzzySets.append(TrapezoidalFuzzySet(points[-2], points[-1], float('inf'), trpzPrm, index))
        return fuzzySets

    @staticmethod
    def createFuzzySetsFromNoStrongPartition(points, trpzPrm):
        assert (len(points)-4)%3 == 0, " Fuzzy Set Builder requires a multiple of three plus 4 " \
                                       "as valid number of points, but %d points have been provided."% len(points)
        fuzzySets = []
        fuzzySets.append(TrapezoidalFuzzySet(-float('inf'), points[0], points[1], trpzPrm=trpzPrm))

        for index in range(1,len(points)-1):
            indexPoints = index*3
            fuzzySets.append(TrapezoidalFuzzySet(points[indexPoints - 1], points[indexPoints], points[indexPoints + 1],trpzPrm=trpzPrm))

        fuzzySets.append(TrapezoidalFuzzySet(points[-2], points[-1], float('inf'),trpzPrm=trpzPrm))
        return fuzzySets

