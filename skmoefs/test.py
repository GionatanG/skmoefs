from __future__ import print_function

import random
import time

import numpy as np
import re
import pickle
import os.path
import pandas as pd
from itertools import product

from platypus.algorithms import *
from multiprocessing import Pool, freeze_support

import matplotlib.pyplot as plt
from scipy.io import arff
from skmoefs.toolbox import MPAES_RCS, load_dataset, normalize, is_object_present, store_object, load_object, milestones
from skmoefs.rcs import RCSInitializer, RCSVariator
from skmoefs.discretization.discretizer_base import fuzzyDiscretization
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing


def test_time():
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    X, y, attributes, inputs, outputs = load_dataset('newthyroid')

    n_features = X.shape[1]
    min_values = np.zeros([n_features])
    max_values = np.zeros([n_features])
    for j in range(n_features):
        min_values[j] = attributes[j][0]
        max_values[j] = attributes[j][1]
        for i in range(len(X[:, j])):
            X[i, j] = (X[i, j] - min_values[j]) / (max_values[j] - min_values[j])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3)

    Amin = 1
    M = 50
    capacity = 32
    divisions = 8

    # VERSION 1
    np.random.seed(seed)
    random.seed(seed)
    variator = RCSVariator()
    discretizer = fuzzyDiscretization(numSet=5)
    initializer = RCSInitializer(discretizer=discretizer)

    mpaes_prova = MPAES_RCS(M=M, Amin=Amin, capacity=capacity,
                            divisions=divisions, variator=variator, initializer=initializer)
    start_time = time.time()
    mpaes_prova.fit(Xtr, ytr, max_evals=50000)
    print('Elapsed time ', time.time() - start_time)
    # ----------------------------------

    # VERSION 2
    np.random.seed(seed)
    random.seed(seed)
    variator = RCSVariator()
    discretizer = fuzzyDiscretization(numSet=5)
    initializer = RCSInitializer(discretizer=discretizer)

    mpaes_prova2 = MPAES_RCS(M=M, Amin=Amin, capacity=capacity,
                             divisions=divisions, variator=variator, initializer=initializer)
    start_time = time.time()
    mpaes_prova2.fit(Xtr, ytr, max_evals=50000)
    print('Elapsed time ', time.time() - start_time)
    # ----------------------------------

def set_rng(seed):
    np.random.seed(seed)
    random.seed(seed)

def make_directory(path):
    try:
        os.stat(path)
    except:
        os.mkdir(path)


def test_fit(dataset, alg, seed):
    path = 'results/' + dataset + '/' + alg + '/'
    make_directory(path)
    set_rng(seed)
    X, y, attributes, inputs, outputs = load_dataset(dataset)
    X_n, y_n = normalize(X, y, attributes)

    Xtr, Xte, ytr, yte = train_test_split(X_n, y_n, test_size=0.3)

    Amin = 1
    M = 100
    capacity = 64
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

    mpaes_rcs_fdt.show_pareto(Xtr, ytr)
    mpaes_rcs_fdt.show_pareto(Xte, yte)
    #mpaes_rcs_fdt.show_pareto(Xtr, ytr, path=base + '_pareto_train')
    #mpaes_rcs_fdt.show_pareto(Xte, yte, path=base + '_pareto_test')
    #mpaes_rcs_fdt.show_pareto_archives(Xtr, ytr, path=base + '_paretoArch_train')
    #mpaes_rcs_fdt.show_pareto_archives(Xte, yte, path=base + '_paretoArch_test')
    mpaes_rcs_fdt.show_model('median', inputs, outputs)








def test_crossval(dataset, alg, seed):
    path = 'results/' + dataset + '/' + alg + '/'
    make_directory(path)
    set_rng(seed)
    X, y, attributes, inputs, outputs = load_dataset(dataset)
    X, y = normalize(X, y, attributes)

    Amin = 1
    M = 100
    capacity = 64
    divisions = 8
    variator = RCSVariator()
    discretizer = fuzzyDiscretization(numSet=5)
    initializer = RCSInitializer(discretizer=discretizer)
    mpaes_rcs_fdt = MPAES_RCS(M=M, Amin=Amin, capacity=capacity,
                             divisions=divisions, variator=variator, initializer=initializer)
    return mpaes_rcs_fdt.cross_val_score(X, y, num_fold=5, seed=seed,
                                  filename=path + str(seed))


def test_multiple():
    datasets = [
        #'appendicitis',
        'ionosphere']
        #'glass', 'haberman',
                #'hayes-roth', 'heart', 'ionosphere', 'iris',
                #'newthyroid']
    algs = ['mpaes22', 'nsga3', 'moead']#, 'gde3', 'epsmoea']
    seeds = (np.arange(6) + 1)
    pool = Pool(6)
    results = []
    r = pool.starmap_async(test_crossval, product(datasets, algs, seeds), callback=results.append)
    r.wait()  # Wait on the results

def import_data_particular(filename):
    matrix = np.loadtxt(filename)
    X = matrix[:,:-1]
    y = matrix[:,-1]
    return X, y


def test_dataset():
    path = '../dataset/d2/d2-'
    #algs = ['mpaes22', 'nsga3', 'moead', 'gde3', 'epsmoea']
    alg = 'mpaes22'

    Amin = 1
    M = 100
    capacity = 64
    divisions = 8
    directory = 'results/d2/'
    columns = ['position', 'fold','seed', 'Acc_tr', 'Acc_ts', 'TRL', 'NRules', 'Precision', 'Recall', 'Fscore']
    
    for fold in range(5):
        print('Fold:' + str(fold + 1))
        Xtr, ytr = import_data_particular(path + str(fold + 1) + '.tra')
        Xts, yts = import_data_particular(path + str(fold + 1) + '.tst')
        

        for seed in range(6):
            set_rng(seed)
            print('Seed:' + str(seed))
            variator = RCSVariator()
            discretizer = fuzzyDiscretization(numSet=5)
            initializer = RCSInitializer(discretizer=discretizer)

            base = directory + 'moefs_' + str(seed) + '_' + str(fold)
            if not is_object_present(base):
                moefs_rcs_fdt = MPAES_RCS(M=M, Amin=Amin, capacity=capacity,
                                          divisions=divisions, variator=variator,
                                          initializer=initializer, moea_type=alg,
                                          objectives=['auc', 'trl'])
                moefs_rcs_fdt.fit(Xtr, ytr, max_evals=50000)
                store_object(moefs_rcs_fdt, base)
            else:
                continue
                moefs_rcs_fdt = load_object(base)

            positions = ['first', 'median', 'last']
            data = {k: [] for k in columns}
            for position in positions:
                classifier = moefs_rcs_fdt[position]
                data['position'].append(position)
                data['fold'].append(fold)
                data['seed'].append(seed)
                data['Acc_tr'].append(classifier.accuracy(Xtr, ytr))
                data['Acc_ts'].append(classifier.accuracy(Xts, yts))
                data['TRL'].append(classifier.trl())
                data['NRules'].append(classifier.num_rules())
                pred_ts = classifier.predict(Xts)
                prec, recall, fscore, _ = metrics.precision_recall_fscore_support(yts, pred_ts,
                                                                                  average='weighted')
                data['Precision'].append(prec)
                data['Recall'].append(recall)
                data['Fscore'].append(fscore)

            df = pd.DataFrame(data, columns=columns)
            df.to_csv('results/d2/data.csv', mode='a', header=False)

def parse_results():
    datasets = ['appendicitis', 'bupa', 'glass', 'haberman',
                'hayes-roth', 'iris', 'heart', 'newthyroid',
                'pima', 'saheart', 'sonar', 'tae', 'vehicle', 'wine']
    algs = ['mpaes22', 'nsga3', 'moead']
    seeds = (np.arange(6) + 1)
    measurements = ['Acc_tr', 'Acc_ts', 'TRL', 'NRules', 'Nsols', 'Precision', 'Recall', 'Fscore']
    positions = ['first', 'median', 'last']
    cols = ['alg', 'dataset'] + [t[0] + '_' + t[1] for t in itertools.product(measurements, positions)]
    stats = {k: [] for k in cols}
    for alg in algs:
        for dataset in datasets:
            data = {k: np.zeros(30) for k in cols}
            for k in range(len(seeds)):

                set_rng(seeds[k])
                scores = test_crossval(dataset, alg, seeds[k])

                for fold in range(5):
                    for i, position in enumerate(positions):
                        for j, meas in enumerate(measurements):
                            data[meas + '_' + position][k*5 + fold] = scores[j, i, fold]
            stats['alg'].append(alg)
            stats['dataset'].append(dataset)
            for k in cols:
                if k != 'alg' and k != 'dataset':
                    stats[k].append(np.mean(data[k]))
    df = pd.DataFrame(stats, columns=cols)
    df.to_csv('csv/stats.csv', mode='a', header=True)

def parse_results_archives():
    datasets = ['appendicitis']
    algs = ['mpaes22']
    seeds = (np.arange(6) + 1)
    measurements = ['Acc_tr', 'Acc_ts', 'TRL', 'NRules', 'Nsols', 'Precision', 'Recall', 'Fscore']
    positions = ['first', 'median', 'last']
    cols = ['alg', 'dataset', 'milestone'] + [t[0] + '_' + t[1] for t in itertools.product(measurements, positions)]
    stats = {k: [] for k in cols}
    for alg in algs:
        for dataset in datasets:
            data = {k: np.full([len(milestones), 30], np.nan) for k in cols}
            for k in range(len(seeds)):

                set_rng(seeds[k])
                scores, scores_archives = test_crossval(dataset, alg, seeds[k])

                for s in range(len(scores_archives)):
                    for fold in range(5):
                        for i, position in enumerate(positions):
                            for j, meas in enumerate(measurements):
                                data[meas + '_' + position][s][k*5 + fold] = scores_archives[s][j, i, fold]
            for s in range(8):
                stats['alg'].append(alg)
                stats['dataset'].append(dataset)
                stats['milestone'].append(milestones[s])
                for k in cols:
                    if k != 'alg' and k != 'dataset' and k != 'milestone':
                        stats[k].append(np.nanmean(data[k][s, :]))
    df = pd.DataFrame(stats, columns=cols)
    df.to_csv('csv/stats_archives.csv', mode='a', header=True)

if __name__=="__main__":
    freeze_support()
    #test_multiple()
    #parse_results()
    parse_results_archives()
