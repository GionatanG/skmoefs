import copy

from skmoefs.frbs import FuzzyRuleBasedClassifier, ClassificationRule
from skmoefs.discretization.discretizer_base import fuzzyDiscretization
from skmoefs import fmdt

import random
import numpy as np

from platypus import Problem, Solution, Binary, Real, Variator



class RCSVariator(Variator):

    def __init__(self, p_crb=0.2, p_cdb=0.5, alpha=0.5, p_mrb=0.1, p_mrb_add=0.5, p_mrb_del=0.2,
                 p_mrb_flip=0.7, p_mdb=0.2, p_cond=0.5):
        super(RCSVariator, self).__init__(2)
        self.p_crb = p_crb
        self.p_cdb = p_cdb
        self.alpha = alpha
        self.p_mrb = p_mrb
        self.p_mrb_add = p_mrb_add
        self.p_mrb_del = p_mrb_del
        self.p_mrb_flip = p_mrb_flip
        self.p_mdb = p_mdb
        self.p_cond = p_cond

    # Remove duplicates from a solution
    def _remove_duplicates(self, solution):
        problem = solution.problem
        rules = set(solution.variables[:2 * problem.M:2])
        for i in range(problem.M - 1):
            if solution.variables[2 * i] not in rules:
                solution.variables[2 * i] = 0
                solution.variables[2 * i + 1][:] = [False for _ in range(problem.F)]
            else:
                rules.remove(solution.variables[2 * i])
        return solution

    def _check_rules_constraints(self, solution):
        problem = solution.problem
        rules = np.unique([v for v in solution.variables[:2 * problem.M:2] if v != 0])
        if len(rules) < problem.Mmin:
            return False

        rules_indexes = [i for i, v in enumerate(solution.variables[:2 * problem.M:2]) if v != 0]
        for i in rules_indexes:
            rule = problem.J[solution.variables[2 * i] - 1, :]
            n_antecedents = 0
            for j in range(problem.F):
                if (rule[j] != 0) and (solution.variables[2 * i + 1][j] != 0):
                    n_antecedents += 1
            if n_antecedents < problem.Amin:
                return False
        return True

    def _sort_rules(self, solution):
        problem = solution.problem
        for i in range(problem.M - 1):
            i_min = i
            for j in range(i + 1, problem.M):
                if (solution.variables[2 * j] != 0) and \
                        ((solution.variables[2 * j] < solution.variables[2 * i_min]) or solution.variables[2 * i] == 0):
                    i_min = j
            if i_min != i:
                solution.variables[2 * i], solution.variables[2 * i_min] = \
                    solution.variables[2 * i_min], solution.variables[2 * i]
                for k in range(problem.F):
                    solution.variables[2 * i + 1][k], solution.variables[2 * i_min + 1][k] = \
                        solution.variables[2 * i_min + 1][k], solution.variables[2 * i + 1][k]
        return solution

    # Add or modify a rule
    def _add_or_modify(self, solution):
        problem = solution.problem
        child_copy = copy.deepcopy(solution)
        old_rule = random.randint(0, problem.M - 1)
        candidates = np.setdiff1d(np.arange(1, problem.Mmax + 1), child_copy.variables[:2 * problem.M:2])
        if len(candidates) > 0:
            new_rule_i = random.choice(candidates)
            indexes = [i for i, v in enumerate(problem.J[new_rule_i - 1, :-1]) if v != 0]
            if len(indexes) >= problem.Amin:
                chosen_indexes = random.sample(indexes, k=random.randint(problem.Amin, len(indexes)))
                child_copy.variables[2 * old_rule] = new_rule_i
                child_copy.variables[2 * old_rule + 1][:] = [(i + 1) in chosen_indexes for i in range(problem.F)]

                if self._check_rules_constraints(child_copy):
                    child_copy = self._remove_duplicates(child_copy)
                    child_copy = self._sort_rules(child_copy)
                    return child_copy
        return solution

    # Delete a rule
    def _del_rule(self, solution):
        problem = solution.problem
        child_copy = copy.deepcopy(solution)
        rules = [i for i, v in enumerate(child_copy.variables[:2 * problem.M:2]) if v != 0]
        if len(rules) > problem.Mmin:
            index = random.choice(rules)
            child_copy.variables[index * 2] = 0
            child_copy.variables[index * 2 + 1][:] = [False for _ in range(problem.F)]
            child_copy = self._sort_rules(child_copy)
            return child_copy
        return solution

    def _db_crossover(self, solution1, solution2):
        problem = solution1.problem
        for i in range(problem.F):
            base = 2 * problem.M + sum(problem.Bfs[:i])
            for j in range(problem.Bfs[i]):
                h1 = solution1.variables[base + j]
                h2 = solution2.variables[base + j]
                h_max = max(h1, h2)
                h_min = min(h1, h2)
                interval = h_max - h_min
                if interval != 0.0:
                    if j == 0:
                        min_h1 = h1 / 2
                        min_h2 = h2 / 2
                    else:
                        min_h1 = h1 - (h1 - solution1.variables[base + j - 1]) / 2
                        min_h2 = h2 - (h2 - solution2.variables[base + j - 1]) / 2

                    if j == problem.Bfs[i] - 1:
                        max_h1 = (1 + h1) / 2
                        max_h2 = (1 + h2) / 2
                    else:
                        max_h1 = h1 + (solution1.variables[base + j + 1] - h1) / 2
                        max_h2 = h2 + (solution2.variables[base + j + 1] - h2) / 2

                    min_h1 = np.clip(min_h1, 0.0, 1.0)
                    min_h2 = np.clip(min_h2, 0.0, 1.0)

                    new_h1 = random.uniform(h_min - self.alpha * interval, h_max + self.alpha * interval)
                    new_h2 = random.uniform(h_min - self.alpha * interval, h_max + self.alpha * interval)

                    new_h1 = np.clip(new_h1, min_h1, max_h1)
                    new_h2 = np.clip(new_h2, min_h2, max_h2)

                    solution1.variables[base + j] = new_h1
                    solution2.variables[base + j] = new_h2
        return solution1, solution2

    def _rb_crossover(self, solution1, solution2):
        problem = solution1.problem
        child_copy1 = copy.deepcopy(solution1)
        child_copy2 = copy.deepcopy(solution2)

        n_rules1 = len([v for v in child_copy1.variables[:2 * problem.M:2] if v != 0])
        n_rules2 = len([v for v in child_copy2.variables[:2 * problem.M:2] if v != 0])
        roMax = min(n_rules1, n_rules2)

        crossover_point = random.randint(1, roMax - 1)

        for i in range(crossover_point):
            child_copy1.variables[2 * i], child_copy2.variables[2 * i] = \
                child_copy2.variables[2 * i], child_copy1.variables[2 * i]
            child_copy1.variables[2 * i + 1][:], child_copy2.variables[2 * i + 1][:] = \
                child_copy2.variables[2 * i + 1][:], child_copy1.variables[2 * i + 1][:]

        if self._check_rules_constraints(child_copy1) and self._check_rules_constraints(child_copy2):
            child_copy1 = self._remove_duplicates(child_copy1)
            child_copy1 = self._sort_rules(child_copy1)
            child_copy2 = self._remove_duplicates(child_copy2)
            child_copy2 = self._sort_rules(child_copy2)
            return child_copy1, child_copy2
        else:
            return solution1, solution2

    def _rb_flipflop_mutation(self, solution):
        problem = solution.problem
        child_copy = copy.deepcopy(solution)
        rule_i = random.choice([i for i, v in enumerate(solution.variables[:2 * problem.M:2]) if v != 0])
        for j in (i for i, v in enumerate(problem.J[solution.variables[2 * rule_i] - 1, :-1]) if v != 0):
            if random.random() < self.p_cond:
                value = child_copy.variables[2 * rule_i + 1][j]
                child_copy.variables[2 * rule_i + 1][j] = not value
        if self._check_rules_constraints(child_copy):
            return child_copy
        return solution

    def _db_random_mutation(self, solution):
        problem = solution.problem
        k = random.randint(0, problem.F - 1)
        if problem.Bfs[k] > 0:
            base = 2 * problem.M + sum(problem.Bfs[:k])
            j = random.randint(0, problem.Bfs[k] - 1) + 1
            domain = [0.0] + solution.variables[base:base + problem.Bfs[k]] + [1.0]
            min_interval = domain[j] - (domain[j] - domain[j - 1]) / 2
            max_interval = domain[j] + (domain[j + 1] - domain[j]) / 2
            solution.variables[base + j - 1] = random.uniform(min_interval, max_interval)
        return solution

    def evolve(self, parents):
        problem = parents[0].problem
        child1 = copy.deepcopy(parents[0])
        child1.evaluated = False
        child2 = copy.deepcopy(parents[1])
        child2.evaluated = False

        # RB CROSSOVER
        if random.random() < self.p_crb:
            child1, child2 = self._rb_crossover(child1, child2)
            force_first_mutation = False
        else:
            force_first_mutation = True
        # DB CROSSOVER
        if random.random() < self.p_cdb:
            child1, child2 = self._db_crossover(child1, child2)

        if force_first_mutation or random.random() < self.p_mrb:
            # FIRST RB MUTATION (Add/Modify)
            if random.random() < self.p_mrb_add:
                child1 = self._add_or_modify(child1)
            # SECOND RB MUTATION (Delete)
            if random.random() < self.p_mrb_del:
                child1 = self._del_rule(child1)
            # THIRD RB MUTATION (Flip-flop)
            if random.random() < self.p_mrb_flip:
                child1 = self._rb_flipflop_mutation(child1)
        # DB MUTATION
        if random.random() < self.p_mdb:
            child1 = self._db_random_mutation(child1)

        if force_first_mutation or random.random() < self.p_mrb:
            # FIRST RB MUTATION (Add/Modify)
            if force_first_mutation or random.random() < self.p_mrb_add:
                child2 = self._add_or_modify(child2)
            # SECOND RB MUTATION (Delete)
            if random.random() < self.p_mrb_del:
                child2 = self._del_rule(child2)
            # THIRD RB MUTATION (Flip-flop)
            if random.random() < self.p_mrb_flip:
                child2 = self._rb_flipflop_mutation(child2)
        # DB MUTATION
        if random.random() < self.p_mdb:
            child2 = self._db_random_mutation(child2)

        children = []
        if problem.check_solution(child1):
            children.append(child1)
        if problem.check_solution(child2):
            children.append(child2)
        return children


class RCSInitializer:

    def __init__(self, discretizer=fuzzyDiscretization(5), tree=fmdt.FMDT(priorDiscretization=True, verbose=True)):
        self.discretizer = discretizer
        self.tree = tree
        self.fTree = None
        self.splits = None
        self.rules = None

    def fit_tree(self, x, y):
        continous = [True] * x.shape[1]
        cPoints = self.discretizer.run(x, continous)
        self.fTree = self.tree.fit(x, y, cPoints=cPoints, continous=continous)
        self.rules = np.array(self.fTree.tree._csv_ruleMine(x.shape[1], []))
        self.rules[:, -1] -= 1
        self.splits = np.array(self.fTree.cPoints)

    def get_splits(self):
        return self.splits

    def get_rules(self):
        return self.rules


class RCSProblem(Problem):

    def __init__(self, Amin, M, J, splits, objectives):

        self.Amin = Amin
        self.M = M
        self.Mmax = J.shape[0]
        self.Mmin = len(set(J[:, -1]))
        self.M = np.clip(self.M, self.Mmin, self.Mmax)
        self.F = J.shape[1] - 1

        self.J = J
        self.initialBfs = splits
        self.G = np.array([len(split) for split in splits])
        self.Bfs = self.G - 2

        self.objectives = objectives
        super(RCSProblem, self).__init__(2 * self.M + sum(self.Bfs), len(objectives))
        self.types[0:2 * self.M:2] = [Real(0, self.M) for _ in range(self.M)]
        self.types[1:2 * self.M:2] = [Binary(self.F) for _ in range(self.M)]
        self.types[2 * self.M:] = [Real(0, 1) for _ in range(sum(self.Bfs))]

        # Maximize objectives by minimizing opposite
        self.directions[:] = [Problem.MINIMIZE for _ in range(len(objectives))]

        self.train_x = None
        self.train_y = None

    def set_training_set(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y

    def decode(self, solution):
        crb = solution.variables[:2 * self.M]
        cdb = solution.variables[2 * self.M:]
        rules = []
        partitions = []
        for j in range(self.F):
            base = sum(self.Bfs[:j])
            full_bfs = np.array([0.0] + cdb[base: base + self.Bfs[j]] + [1.0])
            partitions.append(full_bfs)
        for i in range(self.M):
            km = crb[2 * i]
            if km != 0:
                A = {}
                fuzzyset = {}
                for j in range(self.F):
                    ant = crb[2 * i + 1][j]
                    if (ant and (self.J[km - 1, j] != 0)):
                        fuzzyset[j] = self.J[km - 1, j]
                        index = fuzzyset[j] - 1
                        if index == 0:
                            A[j] = np.array([partitions[j][index], partitions[j][index], partitions[j][index + 1]])
                        elif index == self.G[j] - 1:
                            A[j] = np.array([partitions[j][index - 1], partitions[j][index], partitions[j][index]])
                        else:
                            A[j] = np.array([partitions[j][index - 1], partitions[j][index], partitions[j][index + 1]])
                if len(A) > 0:
                    rule = ClassificationRule(antecedent=A, fuzzyset=fuzzyset, consequent=self.J[km - 1, -1])
                    rules.append(rule)

        classifier = FuzzyRuleBasedClassifier(rules, partitions)
        if (self.train_x is not None) and (self.train_y is not None):
            classifier.compute_weights(self.train_x, self.train_y)
        return classifier

    def evaluate_obj(self, classifier, obj, x=None, y=None):
        if obj == 'auc':
            return 1.0 - (classifier.auc(x, y))
        elif obj == 'accuracy':
            return 1.0 - (classifier.accuracy(x, y))
        elif obj == 'trl':
            return classifier.trl()
        elif obj == 'nrules':
            return classifier.num_rules()

    def evaluate(self, solution):
        classifier = self.decode(solution)
        for i, obj in enumerate(self.objectives):
            solution.objectives[i] = self.evaluate_obj(classifier, obj, self.train_x, self.train_y)


    def _generate_random_rules(self, solution):
        # Choose rules
        n_rules = random.randint(max(self.Mmin, int(self.M * 0.5)), self.M)
        rules = np.zeros([self.M], dtype=int)
        indexes = []
        chosen = np.zeros([n_rules], dtype=int)
        classes = np.unique(self.J[:, -1])
        n_classes = len(classes)
        for j in range(n_classes):
            c = classes[j]
            ind = [i + 1 for i in range(self.J.shape[0]) if self.J[i, -1] == c]
            if len(ind) == 1:
                chosen[j] = ind[0]
            else:
                chosen[j] = ind.pop(random.randrange(len(ind)))
                indexes += ind

        chosen[n_classes:] = random.sample(indexes, k=n_rules - n_classes)
        rules[:n_rules] = np.sort(chosen)
        solution.variables[:2 * self.M:2] = rules

        # Choose antecedents
        for i in range(self.M):
            antecedents = np.ones([self.F], dtype=bool)
            ant_indexes = [i for i, v in enumerate(self.J[rules[i] - 1, :-1]) if v == 0]
            for index in ant_indexes:
                antecedents[index] = False
            solution.variables[2 * i + 1] = antecedents
        return solution

    def _generate_random_fuzzysets(self, solution):
        for i in range(self.F):
            for j in range(self.Bfs[i]):
                solution.variables[2 * self.M + (sum(self.Bfs[:i]) + j)] = self.initialBfs[i][j + 1]

    def random(self):
        solution = Solution(self)
        self._generate_random_rules(solution)
        self._generate_random_fuzzysets(solution)
        return solution

    def check_solution(self, solution):
        crb = np.array(solution.variables[:2 * self.M])
        for i in range(self.M):
            if crb[2 * i] != 0:
                n_antecedents = np.sum(np.logical_and([v != 0 for v in self.J[i, :-1]], crb[2 * i + 1]))
                if n_antecedents < self.Amin:
                    return False
        return sum([v != 0 for v in crb[::2]]) >= self.Mmin
