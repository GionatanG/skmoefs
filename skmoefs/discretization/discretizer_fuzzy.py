#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Mon Sep 26 09:07:41 2016

Python version of the scala FuzzyMDLFilter. 

@author: marcobarsacchi
"""
from __future__ import division, print_function
import bisect
import logging
import concurrent.futures as futures
import os

import numpy as np
from numba import jit


logging.info('Begin')

logger = logging.getLogger('FuzzyMDLFilter')
logger.setLevel(logging.INFO)

class FuzzyMDLFilter(object):

    
    def __init__(self, numClasses, data, label, continous, minImpurity=0.02,
                 minGain=0.000001, threshold = 0, num_bins = 500, ignore=True, ftype="triangular", trpzPrm=0.1):

        """Class for performing fuzzy discretization of the dataset.
        
        Attributes
        ----------
        numClasses: int
            Number of classes in the dataset.
        data: np.array, shape (numSamples, numFeatures)
            N samples, M features
        label: np.array, shape (numSamples)
            Class label for each sample.
        countinous: list, shape(numFeatures)
            True for each continous feature, False for each categorical feature.
        minNumExamples: int
            Minimum number of examples per node.
        minGain: float
            Minimum entropy gain per spit.
        num_bins: int, default =500
            number of bins for the discretization step.
        
        """
        self.numClasses = numClasses
        self.continous = continous
        self.minNumExamples = int(minImpurity*data.shape[0])
        self.data = data
        self.label = label
        self.minGain = minGain
        self.threshold = threshold
        self.num_bins = num_bins
        self.ignore = ignore
        self.trpzPrm = trpzPrm
        self.ftype=ftype

        
    def run(self):
        """
        Parameters
        ----------
        
        Returns
        -------
        List of arrays of candidate splits
        """

        if self.ftype=="triangular":
            logger.info("\nRunning Discretization with fuzzyset " + str(self.ftype))

        if self.ftype=="trapezoidal":
            logger.info("\nRunning Discretization with fuzzyset " + str(self.ftype)+ " and trpzPrm = " + str(self.trpzPrm))
        
        
        self.candidateSplits = self.findCandidateSplits(self.data)
        self.initEntr = np.zeros(self.data.shape[1])
        self.cutPoints = self.findBestSplits(self.data)
        if sum([len(k) for k in self.cutPoints]) == 0: 
            logger.warning('Empty cut points for all the features!')
        
        return self.cutPoints
    
        
    def findCandidateSplits(self, data):
        """Find candidate splits on the dataset.
        
        .. note:
            In the simplest case, the candidate splits are the unique elements
            in the sorted feature array.
        """
        self.N,self.M = data.shape
        num_bins = self.num_bins
        
        
        vector= []
        for k in range(self.M):
            uniques = np.unique(np.sort(data[:,k]))
            if len(uniques)>num_bins:
                vector.append(uniques[np.linspace(0,len(uniques)-1,num_bins, dtype=int)])
            else:
                vector.append(uniques)
        return vector
 
        
        
        
    def findBestSplits(self, data):
        """Find best splits among the data.
        
        """
        
        # Define histogram vectors
        logger.info("BUILDING HISTOGRAMS...")
        self.histograms = []
        for k in range(self.M):
            self.histograms.append(np.zeros(((len(self.candidateSplits[k])+1)*self.numClasses),dtype=int))
        
        # Iterate among features
        for k in range(self.M):
            # Iterate among features
            if self.continous[k]:
                for ind in range(self.N):
                    x = self.simpleHist(data[ind][k], k)
                    self.histograms[k][int(x*self.numClasses +self.label[ind])] +=1
        
        # Histograms built
        
        splits = []
        for k in range(self.M):
            if self.continous[k]:
                if self.threshold == 0:
                    indexCutPoints = self.calculateCutPoints(k, 0 ,
                                                             len(self.histograms[k])-self.numClasses)
                
                else:
                    indexCutPointsIndex = self.calculateCutPointsIndex(k, 0 ,
                                                                       len(self.histograms[k])-self.numClasses)
                    if len(indexCutPointsIndex) != 0:
                        depth,points,_ =  zip(*self.traceback(indexCutPointsIndex))
                        indexCutPoints = sorted(points[:self.threshold])
                    else:
                        indexCutPoints = []
                    
                if len(indexCutPoints) != 0:
                    cutPoints = np.zeros(len(indexCutPoints)+2)
                    cutPoints[0] = self.candidateSplits[k][0]
                    
                    for i in range(len(indexCutPoints)):
                        cSplitIdx = indexCutPoints[i] /self.numClasses
                        if (cSplitIdx > 0 and cSplitIdx < len(self.candidateSplits[k])):
                            cutPoints[i+1] = self.candidateSplits[k][int(cSplitIdx)]
    
                    cutPoints[-1] = self.candidateSplits[k][-1]
                
                    splits.append(cutPoints)
                else:
                    splits.append([])
            else:
                splits.append([])
        return splits
        
    @staticmethod
    def traceback(splitList):
        """ Trace a list of splits back to its original order.
        
        Given a list of split points, the gain value, and its position in the
        splitting "chain". Given by the regular expression T(.[lr]){0,}
        A split in position T.x..[lr] can exist only if exist the corresponding
        father split.
        
        """
        if len(splitList) == 0:
            return []
        base = 'T'
        listValues = []
        k = [k for k,val in enumerate(splitList) if val[2]==base][0]
        # The first split point 'T' is added to the list of selected splits.
        listValues.append(splitList.pop(k))
        available = []
        while splitList:
            # The child of the latest chosen point are inserted in the availability list
            vals = [val for val in splitList if val[2]==(base+'.r') or val[2]==(base+'.l')]
            for val in vals:
                available.append(splitList.pop(splitList.index(val)))
                
            # Choose one of the split point from the available points in the list
            chosen = max(available, key=lambda x: x[0])
            # Update the current chosen point
            base = chosen[2]
            # Insert the chosen point in the list 
            listValues.append(available.pop(available.index(chosen)))
        
        return listValues
        
        
        
    def simpleHist(self, e, fIndex):
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
            return -ind-1
        return ind
        
        
        
    def calculateCutPoints(self, fIndex, first, lastPlusOne, depth=0):
        """ Main iterator
        """
            
        s = np.sum(evalCounts(self.histograms[fIndex], self.numClasses, first, lastPlusOne)) 
        # Evaluating prior cardinality
        if self.ftype=="trapezoidal":
            
            s0s1 = calculatePriorTrapezoidalCardinality(self.histograms[fIndex],
                                                   self.candidateSplits[fIndex],
                                                    self.numClasses, first, lastPlusOne,self.trpzPrm)
        elif self.ftype=="triangular":
            
            s0s1 = calculatePriorTriangularCardinality(self.histograms[fIndex],
                                                   self.candidateSplits[fIndex],
                                                    self.numClasses, first, lastPlusOne)
    
        wPriorFuzzyEntropy = calculateWeightedFuzzyImpurity(s0s1,s, entropy)
        s0s1s2 = np.zeros((3, self.numClasses))
        bestEntropy = wPriorFuzzyEntropy
        bestS0S1S2 = np.zeros_like(s0s1s2)
        bestIndex = -1
        currentEntropy = 0.0
        leftNumInstances = 0
        currSplitIndex = first
        currClassIndex = 0

        while(currSplitIndex < lastPlusOne):
            
            if (leftNumInstances > self.minNumExamples and (s-leftNumInstances) > self.minNumExamples):
                if self.ftype=="trapezoidal":
                    
                    s0s1s2 = calculateNewTrapezoidalCardinality(self.histograms[fIndex], 
                                                                 self.candidateSplits[fIndex],
                                                                 currSplitIndex/self.numClasses,
                                                                 self.numClasses, first, lastPlusOne, self.trpzPrm)
                elif self.ftype=="triangular":
                    
                    s0s1s2 = calculateNewTriangularCardinality(self.histograms[fIndex], 
                                                               self.candidateSplits[fIndex],
                                                                currSplitIndex/self.numClasses,
                                                                self.numClasses, first, lastPlusOne)
                
                currentEntropy = calculateWeightedFuzzyImpurity(s0s1s2, s, entropy)
                

                if currentEntropy < bestEntropy:
                    bestEntropy = currentEntropy
                    bestIndex = currSplitIndex
                    bestS0S1S2 = s0s1s2.copy()

                    
            for currClassIndex in range(0,self.numClasses):
                leftNumInstances += self.histograms[fIndex][currSplitIndex+currClassIndex]
            
            currSplitIndex += self.numClasses
            
        
        gain = wPriorFuzzyEntropy - bestEntropy
        
        if (gain < self.minGain or not self.mdlStopCondition(gain, wPriorFuzzyEntropy, bestS0S1S2, entropy,None)):# (bestwEntroA, besttCountA, bestwEntroB, besttCountB))):
            logger.debug("Feature %d index %d, gain %f REJECTED" %(fIndex, bestIndex, gain))
            return np.array([])
        logger.debug("Feature %d index %d, gain %f ACCEPTED" %(fIndex, bestIndex, gain))
    
        left = self.calculateCutPoints(fIndex, first, bestIndex,depth=depth+1)
        right = self.calculateCutPoints(fIndex, bestIndex, lastPlusOne,depth=depth+1)
        
        indexCutPoints = np.zeros((len(left)+ 1 + len(right)))

        for k in range(len(left)):
            indexCutPoints[k] = left[k]
        indexCutPoints[len(left)] = bestIndex
        for k in range(len(right)):
            indexCutPoints[len(left)+1+k] = right[k]
            
        return indexCutPoints
                    
    def calculateCutPointsIndex(self, fIndex, first, lastPlusOne, depth=0, baseIndex='T'):
        """ Main iterator
        """
        
        s = sum(evalCounts(self.histograms[fIndex], self.numClasses, first, lastPlusOne)) 
        # Evaluating prior cardinality
        if self.ftype=="trapezoidal":
           s0s1 = calculatePriorTrapezoidalCardinality(self.histograms[fIndex],
                                                   self.candidateSplits[fIndex],
                                                    self.numClasses, first, lastPlusOne, self.trpzPrm)
        elif self.ftype=="triangular":
            s0s1 = calculatePriorTriangularCardinality(self.histograms[fIndex],
                                                   self.candidateSplits[fIndex],
                                                    self.numClasses, first, lastPlusOne)
        
        wPriorFuzzyEntropy = calculateWeightedFuzzyImpurity(s0s1,s, entropy)
        if depth == 0:
            self.initEntr[fIndex] = wPriorFuzzyEntropy
        s0s1s2 = np.zeros((3, self.numClasses))
        bestEntropy = wPriorFuzzyEntropy
        bestS0S1S2 = np.zeros_like(s0s1s2)
        bestIndex = -1
        currentEntropy = 0.0
        leftNumInstances = 0
        currSplitIndex = first
        currClassIndex = 0

        
        while(currSplitIndex < lastPlusOne):
            
            if (leftNumInstances > self.minNumExamples and (s-leftNumInstances) > self.minNumExamples):
                s0s1s2 = calculateNewTriangularCardinality(self.histograms[fIndex], 
                                                           self.candidateSplits[fIndex],
                                                            currSplitIndex/self.numClasses,
                                                            self.numClasses, first, lastPlusOne)
                currentEntropy = calculateWeightedFuzzyImpurity(s0s1s2, s, entropy)
                

                if currentEntropy < bestEntropy:
                    bestEntropy = currentEntropy
                    bestIndex = currSplitIndex
                    bestS0S1S2 = s0s1s2.copy()

                    
            for currClassIndex in range(0,self.numClasses):
                leftNumInstances += self.histograms[fIndex][currSplitIndex+currClassIndex]
            
            currSplitIndex += self.numClasses
            

        gain = wPriorFuzzyEntropy - bestEntropy

        
        if depth<5 and self.ignore:
            if (gain < self.minGain):
                logger.debug("Feature %d index %d, gain %f REJECTED" %(fIndex, bestIndex, gain))
                return []
            logger.debug("Feature %d index %d, gain %f ACCEPTED" %(fIndex, bestIndex, gain))
        else:
            if (gain < self.minGain or not self.mdlStopCondition(gain,
                                                                 wPriorFuzzyEntropy, bestS0S1S2, entropy,None)):
                logger.debug("Feature %d index %d, gain %f REJECTED" %(fIndex, bestIndex, gain))
                return []
            logger.debug("Feature %d index %d, gain %f ACCEPTED" %(fIndex, bestIndex, gain))
        
        left = self.calculateCutPointsIndex(fIndex, first, bestIndex,depth=depth+1,baseIndex = baseIndex+'.l')
        right = self.calculateCutPointsIndex(fIndex, bestIndex, lastPlusOne,depth=depth+1,baseIndex = baseIndex+'.r')
        
        
        return left +[(gain*s/self.data.shape[0],bestIndex,baseIndex)] + right
        
    def mdlStopCondition(self, gain, priorEntropy, bestCounts, impurity, bestPartition):
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
        bestPartition: tuple, (float, int, float, int)
            tuple of entropy and class counts for the new partitions.
        
        """
        bestClassCounts = np.zeros((len(bestCounts)))
        for i in range(len(bestCounts)):
            for j in range(len(bestCounts[i])):
                if bestCounts[i,j]> 0.0:
                    bestClassCounts[i] += 1
        totalCounts = bestCounts.sum()
        
        totalClassCounts = 0
        pivot = bestCounts.sum(axis = 0)
        for k in pivot:
            if k > 0.0:
                totalClassCounts += 1
        
        # Formulation from Armando
        delta = np.log2(np.power(3,totalClassCounts)-2) - totalClassCounts * priorEntropy + sum([ bestClassCounts[k]*entropy(bestCounts[k], sum(bestCounts[k])) for k in range(len(bestCounts))]) 
        
        #delta = np.log2(np.power(3,totalClassCounts)-2) - totalClassCounts * priorEntropy + \
        #                bestPartition[0]*bestPartition[1]+bestPartition[2]*bestPartition[3]
        return gain > ((np.log2(totalCounts-1) + delta) / (totalCounts))
        
        
@jit        
def entropy(counts, totalCount):
    """Evaluate entropy.
    
    """
    if totalCount == 0:
        return 0
    
    numClasses = len(counts)
    impurity = 0.0
    classIndex = 0
    while(classIndex < numClasses):
        classCount = counts[classIndex]
        if classCount != 0:
            freq = classCount / totalCount
            if freq != 0:
                impurity -= freq * np.log2(freq)
        classIndex += 1
    
    return impurity

@jit    
def calculateNewTriangularCardinality(hist, candidateSplits, indexOfCandidateSplit, numClasses, 
                                      first, lastPlusOne):
    """Partition cardinality for a 3 triangular fuzzy set partition.
    
    Parameters
    ----------
    hist: np.array, shape(numCandidateSplits*numClasses)
        histograms
    candidateSplits: np.array, shape(numCandidateSplits)
        array of candidate split points
    indexOfCandidateSplit: int
        index of current candidate split to evaluate
    numClasses: int
        number of classes in the current partition
    first: int
        first index of the partition
    lastPlusOne: int
        last index of the partition (plus one)
        
    Returns
    -------
    a 3 by M matrix, where M is the number of classes, 
    where matrix[i,j] contains the cardinality of set i for the class j.
    
    """
    
    s0s1s2 = np.zeros((3, numClasses))
    minimum = candidateSplits[first//numClasses]
    peak = candidateSplits[int(indexOfCandidateSplit)]
    diff = peak - minimum
    uSi = 0.0
    xi = 0.0
    
    for i in range(int(first/numClasses), int(indexOfCandidateSplit+1)):
        xi = candidateSplits[i]
        uSi = (xi-minimum)/diff
        for j in range(0, numClasses):
            s0s1s2[0][j] += hist[i*numClasses+j] * (1-uSi)
            s0s1s2[1][j] += hist[i*numClasses+j] * uSi
    diff = candidateSplits[lastPlusOne//numClasses-1] - peak 

    for i in range(int(indexOfCandidateSplit+1), int(lastPlusOne/numClasses)):
        xi = candidateSplits[i]
        uSi = (xi-peak)/diff
        for j in range(0, numClasses):
            s0s1s2[1][j] += hist[i*numClasses+j] * (1-uSi)
            s0s1s2[2][j] += hist[i*numClasses+j] * uSi
    
    return s0s1s2
    
@jit    
def calculateWeightedFuzzyImpurity(partition, partitionCardinality, impurity):
    """Perform weighted fuzzy impurity evaluation on a fuzzy partition.
    
    Parameters
    ----------
    Partition: np.array, shape(numPartitions, numClasses)
        Partition with numPartitions subpartitions and numClasses classes.
    partitionCardinality: float
        cardinality of the partition
    impurity: function
        evaluate impurity
    Returns
    -------
    Weighted impurity.    
    """
    
    summedImp = 0.0
    
    for k in range(len(partition)):
        summedImp += np.sum(partition[k])/ partitionCardinality * impurity(partition[k], np.sum(partition[k])) 
    
    return summedImp

@jit        
def evalCounts(hist, numClasses, first, lastPlusOne):
    """ Number of counts.
    
    """
    counts = np.zeros((numClasses))
    index = first
    while (index<lastPlusOne):
        for k in range(len(counts)):
            counts[k] += hist[index+k]
        index += numClasses
    return counts


@jit
def calculatePriorTriangularCardinality(hist, candidateSplits, numClasses, first, lastPlusOne):
    """Prior triangular cardinality.
    """
   
    s0s1 = np.zeros((2, numClasses))
    minimum = candidateSplits[first // numClasses]
    diff = candidateSplits[lastPlusOne // numClasses - 1] - minimum
    uS1 = 0.0
    xi = 0.0
    for i in range(int(first / numClasses), int(lastPlusOne / numClasses)):
        xi = candidateSplits[i]
        uS1 = 0.0
        if diff != 0.:
            uS1 = (xi - minimum) / diff
        for j in range(0, numClasses):
            s0s1[0][j] += hist[i * numClasses + j] * (1 - uS1)
            s0s1[1][j] += hist[i * numClasses + j] * uS1

    return s0s1

#--------CORRADO--------------------------------------------------------------
def calculateNewTrapezoidalCardinality(hist, candidateSplits, indexOfCandidateSplit, numClasses, 
                                      first, lastPlusOne, trpzPrm=0.2):
    """Partition cardinality for a 3 Trapezoidal fuzzy set partition.
    
    Parameters
    ----------
    hist: np.array, shape(numCandidateSplits*numClasses)
        histograms
    candidateSplits: np.array, shape(numCandidateSplits)
        array of candidate split points
    indexOfCandidateSplit: int
        index of current candidate split to evaluate
    numClasses: int
        number of classes in the current partition
    first: int
        first index of the partition
    lastPlusOne: int
        last index of the partition (plus one)
     trpzPrm: double
            percentuale dell' intervallo in cui la membership function varrà 1
    Returns
    -------
    a 3 by M matrix, where M is the number of classes, 
    where matrix[i,j] contains the cardinality of set i for the class j.
    
    """
    s0s1s2 = np.zeros((3, numClasses))
    minimum = candidateSplits[first//numClasses]
    split = candidateSplits[int(indexOfCandidateSplit)]
    uSi = 0.0
    xi = 0.0
    
    #a sx dello split point
    diff1 = split-minimum 
    plateau1 = diff1*trpzPrm
    slope1=diff1*(1-2*trpzPrm)
    
    for i in range(int(first/numClasses), int(indexOfCandidateSplit+1)):
        xi = candidateSplits[i]
        #Compute uSi
        uSi = (xi-minimum-plateau1)/slope1
        uSi= max(0, uSi) 
        uSi = min (1,uSi)
        #uSi Computed
        for j in range(0, numClasses):
            s0s1s2[0][j] += hist[i*numClasses+j] * (1-uSi)
            s0s1s2[1][j] += hist[i*numClasses+j] * uSi
    
    # a dx dello split point
    diff2 = candidateSplits[lastPlusOne//numClasses-1] - split 
    plateau2 = diff2*trpzPrm
    slope2=diff2*(1-2*trpzPrm)
    
    for i in range(int(indexOfCandidateSplit+1), int(lastPlusOne/numClasses)):
        xi = candidateSplits[i]
        #Compute uSi
        uSi = (xi-split-plateau2)/slope2
        uSi= max(0, uSi) 
        uSi = min (1,uSi)
        #uSi Computed
        for j in range(0, numClasses):
            s0s1s2[1][j] += hist[i*numClasses+j] * (1-uSi)
            s0s1s2[2][j] += hist[i*numClasses+j] * uSi
    
    return s0s1s2
#------------------------------------------------

def calculatePriorTrapezoidalCardinality(hist, candidateSplits, numClasses, first, lastPlusOne, trpzPrm=0.2):
    """Prior trapezoidal cardinality.
    """

    
    s0s1 = np.zeros((2, numClasses))
    minimum = candidateSplits[first // numClasses]
    diff = candidateSplits[lastPlusOne // numClasses - 1] - minimum
    plateau = diff*trpzPrm
    slope=diff*(1-2*trpzPrm)
    uS1 = 0.0
    xi = 0.0
    for i in range(int(first / numClasses), int(lastPlusOne / numClasses)):
        xi = candidateSplits[i]
        uS1 = 0.0
        if slope!= 0.:
             uS1 = (xi-minimum-plateau)/slope
             uS1= max(0, uS1) 
             uS1 = min (1,uS1)
        for j in range(0, numClasses):
            s0s1[0][j] += hist[i * numClasses + j] * (1 - uS1)
            s0s1[1][j] += hist[i * numClasses + j] * uS1
            

    return s0s1


# MULTIPROCESSING STUFF
def parallel_FuzzyMDLFilter(numClasses, data, label, continous, minImpurity=0.02,
                 minGain=0.000001, threshold = 0, num_bins = 500, ignore=True, ftype="triangular", trpzPrm=0.1):
    try:
        num_cores = os.cpu_count()
    except:
        num_cores = 4
    num_cores = min(num_cores, data.shape[1])

    indexes = np.linspace(0,data.shape[1],num_cores+1,dtype=int)
    executor = futures.ProcessPoolExecutor()
    with futures.ProcessPoolExecutor() as executor:
        to_do = []
        for k in range(len(indexes)-1):
            future = executor.submit(FuzzyMDLFilter_run,*(k,numClasses,data[:,indexes[k]:indexes[k+1]],label,continous[indexes[k]:indexes[k+1]]),
            **dict(minImpurity=0.02,
                 minGain=0.000001, threshold = 0, num_bins = 500, ignore=True, ftype="triangular", trpzPrm=0.1))
            to_do.append(future)
            msg = 'Scheduled for {}: {}'
            logger.info(msg.format(k, future))
        results = {}
        for future in futures.as_completed(to_do):
            res = future.result()
            msg = '{} result: {!r}'
            logger.info(msg.format(future, res))
            results[res[0]] = res[1]

        ordered_results = []
        for key in sorted(list(results.keys())):
            ordered_results += results[key]
        return ordered_results

def FuzzyMDLFilter_run(index,numClasses, data, label, continous, minImpurity=0.02,
                 minGain=0.000001, threshold = 0, num_bins = 500, ignore=True, ftype="triangular", trpzPrm=0.1):

    return index,FuzzyMDLFilter(numClasses, data, label, continous, minImpurity=0.02,
                 minGain=0.000001, threshold = 0, num_bins = 500, ignore=True, ftype="triangular", trpzPrm=0.1).run()

