from __future__ import print_function
import numpy as np
import bisect
import logging

logger = logging.getLogger('CrispMDLFilter')
logger.setLevel(logging.INFO)


class CrispMDLFilter(object):
    def __init__(self, numClasses, data, label, continous, minNumExamples=2,
                 minGain=0.01):

        """Class for performing crisp discretization of the dataset.

        Attributes
        ----------
        numClasses: int
            Number of classes in the dataset.
        data: np.array, shape (numSamples, numFeatures)
            N samples, M features
        label: np.array, shape (numSamples)
            Class label for each sample.
        continous: list, shape(numFeatures)
            True for each continous feature, False for each categorical feature.
        minNumExamples: int
            Minimum number of examples per node.
        minGain: float
            Minimum entropy gain per spit.

        """
        self.numClasses = numClasses
        self.continous = continous
        self.minNumExamples = minNumExamples
        self.data = data
        self.label = label
        self.minGain = minGain
        self.candidateSplits = None
        self.cutPoints = None

    def run(self):
        """
        Parameters
        ----------

        Returns
        -------
        Array of arrays of candidate splits
        """

        self.candidateSplits = self.__findCandidateSplits(self.data)

        self.cutPoints = self.__findBestSplits(self.data)

        return self.cutPoints

    def __findCandidateSplits(self, data):
        """Find candidate splits on the dataset.

        .. note:
            In the simplest case, the candidate splits are the unique elements
            in the sorted feature array.
        """
        self.N, self.M = data.shape

        return [np.unique(np.sort(data[:, k])) for k in range(self.M)]

    def __findBestSplits(self, data):
        """Find best splits among the data.

        """

        # Define histogram vectors
        logger.debug("BUILDING HISTOGRAMS...")
        self.histograms = []
        for k in range(self.M):
            self.histograms.append(np.zeros(((len(self.candidateSplits[k]) + 1) * self.numClasses), dtype=int))

        # Iterate among features
        for k in range(self.M):
            # Iterate among features
            if self.continous[k]:
                for ind in range(self.N):
                    x = self.__simpleHist(data[ind][k], k)
                    self.histograms[k][int(x * self.numClasses + self.label[ind])] += 1

        # Histograms built

        splits = []
        for k in range(self.M):
            indexCutPoints = self.__calculateCutPoints(k, 0, len(self.histograms[k]) - self.numClasses)
            if len(indexCutPoints != 0):
                cutPoints = np.zeros(len(indexCutPoints))

                for i in range(len(indexCutPoints)):
                    cSplitIdx = int(indexCutPoints[i] / self.numClasses)
                    if (cSplitIdx > 0 and cSplitIdx < len(self.candidateSplits[k])):
                        cutPoints[i] = self.candidateSplits[k][cSplitIdx]

                splits.append(cutPoints)
            else:
                splits.append([])
        return splits

    def __simpleHist(self, e, fIndex):
        """ Simple binary histogram function.

        Parameters
        ----------
        e: float
            point to locate in the histogram
        fIndex: int
            feature index

        Returns
        -------
        Index of e in the histogram of feature fIndex.
        """
        ind = bisect.bisect_left(self.candidateSplits[fIndex], e)
        if (ind < 0):
            return -ind - 1
        return ind

    def __calculateCutPoints(self, fIndex, first, lastPlusOne):
        """ Main iterator
        """

        counts = np.zeros((2, self.numClasses), dtype=float)
        counts[1, :] = evalCounts(self.histograms[fIndex], self.numClasses, first, lastPlusOne)
        totalCounts = counts[1, :].sum()

        if totalCounts < self.minNumExamples:
            return np.array([])

        priorCounts = counts[1, :].copy()
        priorEntropy = entropy(priorCounts, totalCounts)

        bestEntropy = priorEntropy

        bestCounts = np.zeros((2, self.numClasses), dtype=float)
        bestIndex = -1
        leftNumInstances = 0
        currSplitIndex = first

        while (currSplitIndex < lastPlusOne):

            if (leftNumInstances > self.minNumExamples and (totalCounts - leftNumInstances) > self.minNumExamples):
                leftImpurity = entropy(counts[0, :], leftNumInstances)
                rightImpurity = entropy(counts[1, :], totalCounts - leftNumInstances)
                leftWeight = float(leftNumInstances) / totalCounts
                rightWeight = float(totalCounts - leftNumInstances) / totalCounts
                currentEntropy = leftWeight * leftImpurity + rightWeight * rightImpurity

                if currentEntropy < bestEntropy:
                    bestEntropy = currentEntropy
                    bestIndex = currSplitIndex
                    bestCounts = counts.copy()

            for currClassIndex in range(0, self.numClasses):
                leftNumInstances += self.histograms[fIndex][currSplitIndex + currClassIndex]
                counts[0][currClassIndex] += self.histograms[fIndex][currSplitIndex + currClassIndex]
                counts[1][currClassIndex] -= self.histograms[fIndex][currSplitIndex + currClassIndex]

            currSplitIndex += self.numClasses

        gain = priorEntropy - bestEntropy
        if (gain < self.minGain or not self.__mdlStopCondition(gain, priorEntropy, bestCounts, entropy)):
            logger.debug("Feature %d index %d, gain %f REJECTED" % (fIndex, bestIndex, gain))
            return np.array([])
        logger.debug("Feature %d index %d, gain %f ACCEPTED" % (fIndex, bestIndex, gain))

        left = self.__calculateCutPoints(fIndex, first, bestIndex)
        right = self.__calculateCutPoints(fIndex, bestIndex, lastPlusOne)

        indexCutPoints = np.zeros((len(left) + 1 + len(right)))

        for k in range(len(left)):
            indexCutPoints[k] = left[k]
        indexCutPoints[len(left)] = bestIndex
        for k in range(len(right)):
            indexCutPoints[len(left) + 1 + k] = right[k]

        return indexCutPoints

    def __mdlStopCondition(self, gain, priorEntropy, bestCounts, impurity):
        """mdlStopCondition evaluation.

        Parameters
        ----------
        gain: double
            entropy gain for the best split.
        priorEntropy: double
            entropy for the partition prior to the split
        bestCounts: np.array, shape(numPartitions, numClasses)
            counts for the best split, divided by classes
        impurity: function caller
            impurity function to evaluate

        """
        leftClassCounts = 0
        rightClassCounts = 0
        totalClassCounts = 0
        for i in range(len(bestCounts[0, :])):
            if bestCounts[0, i] > 0.0:
                leftClassCounts += 1
                totalClassCounts += 1
                if bestCounts[1, i] > 0.0:
                    rightClassCounts += 1
            elif bestCounts[1, i] > 0:
                rightClassCounts += 1
                totalClassCounts += 1

        totalCounts = bestCounts.sum()
        delta = np.log2(np.power(3, totalClassCounts) - 2) - \
                totalClassCounts * priorEntropy + \
                leftClassCounts * impurity(bestCounts[0, :], bestCounts[0, :].sum()) + \
                rightClassCounts * impurity(bestCounts[1, :], bestCounts[1, :].sum())
        return gain > ((np.log2(totalCounts - 1) + delta) / (totalCounts))


def entropy(counts, totalCount):
    """Evaluate entropy.

    """
    if totalCount == 0:
        return 0

    numClasses = len(counts)
    impurity = 0.0
    classIndex = 0
    while (classIndex < numClasses):
        classCount = counts[classIndex]
        if classCount != 0:
            freq = classCount / totalCount
            if freq != 0:
                impurity -= freq * np.log2(freq)
        classIndex += 1

    return impurity


def evalCounts(hist, numClasses, first, lastPlusOne):
    """ Number of counts.

    """
    counts = np.zeros((numClasses))
    index = first
    while (index < lastPlusOne):
        for k in range(len(counts)):
            counts[k] += hist[index + k]
        index += numClasses
    return counts