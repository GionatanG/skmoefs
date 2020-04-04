import abc

class FuzzySet(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def membershipDegree(self, xi):
        pass

    @abc.abstractmethod
    def isInSupport(self, xi):
        pass

    @abc.abstractmethod
    def isFirstOfPartition(self):
        pass

    @abc.abstractmethod
    def isLastOfPartition(self):
        pass