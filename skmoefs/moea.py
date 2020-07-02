from __future__ import print_function

import copy
import random

import numpy as np
from platypus import AbstractGeneticAlgorithm, ParetoDominance, \
    AdaptiveGridArchive, Generator, Selector, NSGAII, NSGAIII, GDE3, SPEA2, IBEA, MOEAD, EpsMOEA

from skmoefs.rcs import RCSVariator

milestones = [500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000, 75000, 100000]

class MOEAGenerator(Generator):
    """
    Class for generating the initial population of a Multi-Objective GA
    """
    def __init__(self):
        super(MOEAGenerator, self).__init__()
        self.counter = 0

    def generate(self, problem):
        self.counter += 1
        solution = problem.random()

        return solution

class RandomSelector(Selector):
    """
    Class for randomly select an individual from the population
    """
    def __init_(self):
        super(RandomSelector, self).__init_()

    def select_one(self, population):
        return random.choice(population)

class MPAES2_2(AbstractGeneticAlgorithm):
    """
    M-PEAS(2+2) algorithm class
    """
    def __init__(self,
                 problem,
                 variator=None,
                 capacity=32,
                 divisions=8,
                 generator=MOEAGenerator(),
                 **kwargs):
        """

        :param problem: the problem object which is responsible of creating and evaluating a solution.
        :param variator: the genetic operator which can comprises elements of cross-over and mutation
        :param capacity: the maximum number of solutions that can be held by the archive throughout
            the evolution
        :param divisions: the number of divisions for the grid within the archive. This value
            influences how a new solution will replace an existing one
        :param generator: a MOEA generator to initialize the population/archive
        :param kwargs: ...
        """

        super(MPAES2_2, self).__init__(problem, population_size=2, generator=generator, **kwargs)

        self.variator = variator
        self.dominance = ParetoDominance()
        self.archive = AdaptiveGridArchive(capacity, problem.nobjs, divisions)
        # From time to time we take snapshots of the archive for debug purposes
        self.snapshots = []

    def step(self):
        """
        Execute a new era
        """
        if self.nfe == 0:
            self.initialize()
            self.result = self.archive
        else:
            self.iterate()
            self.result = self.archive

    def initialize(self):
        """
        Initialize the algorithm
        """
        super(MPAES2_2, self).initialize()
        self.archive += self.population

        if self.variator is None:
            self.variator = RCSVariator()

    def iterate(self):
        if (self.nfe % 100) == 0:
            print('Fitness evaluations ', self.nfe)
        if self.nfe >= milestones[len(self.snapshots)]:
            # take a snapshot
            self.snapshots.append(copy.deepcopy(self.archive))

        archive_size = len(self.archive)
        if archive_size <= 2:
            # archive is not big enough to choose 2 parents
            self.population[0] = self.archive[0]
            self.population[1] = self.archive[archive_size - 1]
        else:
            # randomly choose the current 2 parents
            new_parents = random.sample(list(np.arange(archive_size)), k=2)
            self.population[0] = self.archive[new_parents[0]]
            self.population[1] = self.archive[new_parents[1]]

        # generate 2 new children
        children = self.variator.evolve([self.population[0], self.population[1]])

        if len(children) > 0:
            # evaluate fitness function for each new generated child
            self.evaluate_all(children)
            for child in children:
                flag1 = self.dominance.compare(self.population[0], child)
                flag2 = self.dominance.compare(self.population[1], child)
                if (flag1 >= 0) and (flag2 >= 0):
                    # If children are not completely dominated,
                    # then try to add them into the archive
                    self.archive.add(child)


class NSGAIIS(NSGAII):
    """
    Extended version of NSGA2 algorithm which support snapshots
    """

    def initialize(self):
        self.snapshots = []
        super(NSGAIIS, self).initialize()


    def iterate(self):
        if (self.nfe % 100) == 0:
            print('Fitness evaluations ', self.nfe)
        if len(self.snapshots) < len(milestones) and self.nfe >= milestones[len(self.snapshots)]:
            print('new milestone at ', str(self.nfe))
            self.snapshots.append(copy.deepcopy(self.archive))
        super(NSGAIIS, self).iterate()

class NSGAIIIS(NSGAIII):
    """
        Extended version of NSGA3 algorithm which support snapshots
    """

    def initialize(self):
        self.snapshots = []
        super(NSGAIIIS, self).initialize()

    def iterate(self):
        if (self.nfe % 100) == 0:
            print('Fitness evaluations ', self.nfe)
        if len(self.snapshots) < len(milestones) and self.nfe >= milestones[len(self.snapshots)]:
            print('new milestone at ', str(self.nfe))
            self.snapshots.append(copy.deepcopy(self.population))
        super(NSGAIIIS, self).iterate()

class SPEA2S(SPEA2):
    """
        Extended version of SPEA2 algorithm which support snapshots
    """

    def initialize(self):
        self.snapshots = []
        super(SPEA2S, self).initialize()

    def iterate(self):
        if (self.nfe % 100) == 0:
            print('Fitness evaluations ', self.nfe)
        if len(self.snapshots) < len(milestones) and self.nfe >= milestones[len(self.snapshots)]:
            print('new milestone at ', str(self.nfe))
            self.snapshots.append(copy.deepcopy(self.population))
        super(SPEA2S, self).iterate()

class GDE3S(GDE3):
    """
        Extended version of GDE2 algorithm which support snapshots
    """

    def initialize(self):
        self.snapshots = []
        super(GDE3S, self).initialize()

    def iterate(self):
        if (self.nfe % 100) == 0:
            print('Fitness evaluations ', self.nfe)
        if len(self.snapshots) < len(milestones) and self.nfe >= milestones[len(self.snapshots)]:
            print('new milestone at ', str(self.nfe))
            self.snapshots.append(copy.deepcopy(self.population))
        super(GDE3S, self).iterate()

class IBEAS(IBEA):
    """
        Extended version of IBEA algorithm which support snapshots
    """

    def initialize(self):
        self.snapshots = []
        super(IBEAS, self).initialize()

    def iterate(self):
        if (self.nfe % 100) == 0:
            print('Fitness evaluations ', self.nfe)
        if len(self.snapshots) < len(milestones) and self.nfe >= milestones[len(self.snapshots)]:
            print('new milestone at ', str(self.nfe))
            self.snapshots.append(copy.deepcopy(self.population))
        super(IBEAS, self).iterate()

class MOEADS(MOEAD):
    """
        Extended version of MOEAD algorithm which support snapshots
    """

    def initialize(self):
        self.snapshots = []
        super(MOEADS, self).initialize()

    def iterate(self):
        if (self.nfe % 100) == 0:
            print('Fitness evaluations ', self.nfe)
        if len(self.snapshots) < len(milestones) and self.nfe >= milestones[len(self.snapshots)]:
            print('new milestone at ', str(self.nfe))
            self.snapshots.append(copy.deepcopy(self.population))
        super(MOEADS, self).iterate()

class EpsMOEAS(EpsMOEA):
    """
        Extended version of Epsilon-MOEA algorithm which support snapshots
    """

    def initialize(self):
        self.snapshots = []
        super(EpsMOEAS, self).initialize()

    def iterate(self):
        if (self.nfe % 100) == 0:
            print('Fitness evaluations ', self.nfe)
        if len(self.snapshots) < len(milestones) and self.nfe >= milestones[len(self.snapshots)]:
            print('new milestone at ', str(self.nfe))
            self.snapshots.append(copy.deepcopy(self.archive))
        super(EpsMOEAS, self).iterate()