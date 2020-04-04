from __future__ import print_function

import numpy as np
import os.path

from platypus.algorithms import *

from skmoefs.toolbox import MPAES_RCS, load_dataset, normalize, is_object_present, store_object, load_object
from skmoefs.rcs import RCSInitializer, RCSVariator
from skmoefs.discretization.discretizer_base import fuzzyDiscretization
from sklearn.model_selection import train_test_split


def set_rng(seed):
    np.random.seed(seed)
    random.seed(seed)

def make_directory(path):
    try:
        os.stat(path)
    except:
        os.mkdir(path)

def test1():
    X, y, attributes, inputs, outputs = load_dataset('newthyroid')
    X_n, y_n = normalize(X, y, attributes)
    Xtr, Xte, ytr, yte = train_test_split(X_n, y_n, test_size=0.3)

    my_moefs = MPAES_RCS(capacity=32, variator=RCSVariator(), initializer=RCSInitializer())
    my_moefs.fit(Xtr, ytr, max_evals=10000)

    my_moefs.show_pareto()
    my_moefs.show_pareto(Xte, yte)
    my_moefs.show_model('median', inputs=inputs, outputs=outputs)

def test2():
    X, y, attributes, inputs, outputs = load_dataset('newthyroid')
    X_n, y_n = normalize(X, y, attributes)
    my_moefs = MPAES_RCS(variator=RCSVariator(), initializer=RCSInitializer())
    my_moefs.cross_val_score(X_n, y_n, num_fold=5)


def test3(dataset, alg, seed):
    path = 'results/' + dataset + '/' + alg + '/'
    make_directory(path)
    set_rng(seed)
    X, y, attributes, inputs, outputs = load_dataset(dataset)
    X_n, y_n = normalize(X, y, attributes)

    Xtr, Xte, ytr, yte = train_test_split(X_n, y_n, test_size=0.3)

    Amin = 1
    M = 100
    capacity = 32
    divisions = 8
    variator = RCSVariator()
    discretizer = fuzzyDiscretization(numSet=5)
    initializer = RCSInitializer(discretizer=discretizer)
    base = path + 'moefs_' + str(seed)
    if not is_object_present(base):
        mpaes_rcs_fdt = MPAES_RCS(M=M, Amin=Amin, capacity=capacity,
                                  divisions=divisions, variator=variator,
                                  initializer=initializer, moea_type=alg,
                                  objectives=['accuracy', 'trl'])
        mpaes_rcs_fdt.fit(Xtr, ytr, max_evals=1000)
        store_object(mpaes_rcs_fdt, base)
    else:
        mpaes_rcs_fdt = load_object(base)

    mpaes_rcs_fdt.show_pareto()
    mpaes_rcs_fdt.show_pareto(Xte, yte)
    mpaes_rcs_fdt.show_pareto_archives(Xte, yte)
    mpaes_rcs_fdt.show_model('median', inputs, outputs)


if __name__=="__main__":
    test1()
    test2()
    test3('iris', 'nsga2', 2)