.. -*- mode: rst -*-

SKMoefs
============

SKMoefs is a Python module for machine learning specifically built for
Multi-Objective Evolutionary Fuzzy Systems. It takes inspiration by the Scikit-Learn principles and their classes 
are all ScikitLearn's estimators. In addition, it builds upon Platypus, a library that deals with multi-objective algorithms.


Installation
------------

Dependencies
~~~~~~~~~~~~

skmoefs requires:

- Python (>= 3.6)
- NumPy
- SciPy
- Numba
- Matplotlib
- Scikit-Learn
- Platypus

In order to use the library, it is necessary to have the above dependencies installed.

    pip install numpy, scipy, numba, matplotlib, scikit-learn, platypus-opt

Source code
~~~~~~~~~~~

You can check the latest sources with the command::

    git clone https://github.com/GionatanG/skmoefs.git

Testing
~~~~~~~

	You can run some tests by executing

	python example.py

Examples
~~~~~~~~

The simplest example is shown below. 


	from platypus.algorithms import *

	from skmoefs.toolbox import MPAES_RCS, load_dataset, normalize
	from skmoefs.rcs import RCSInitializer, RCSVariator
	from skmoefs.discretization.discretizer_base import fuzzyDiscretization
	from sklearn.model_selection import train_test_split


	X, y, attributes, inputs, outputs = load_dataset('iris')
    X_n, y_n = normalize(X, y, attributes)
    Xtr, Xte, ytr, yte = train_test_split(X_n, y_n, test_size=0.3)

    my_moefs = MPAES_RCS(variator=RCSVariator(), initializer=RCSInitializer())
    my_moefs.fit(Xtr, ytr, max_evals=10000)

    my_moefs.show_pareto()
    my_moefs.show_pareto(Xte, yte)
    my_moefs.show_model('median', inputs=inputs, outputs=outputs)

The program load the IRIS dataset from the built-in datasets and normalize the matrix X. Indeed, 
the input data should be in the form 

- X : real NumPy matrix NxM (number of samples x number of features), with 0 <= X[i,j] <= 1 for every i,j
- y : integer Numpy vector Nx1. In particular, if number of classes is C then y[i] belongs to {1, 2, 3, ...., C} for every i

After the normalization, the script splits the dataset into training and testing. 
It defines and train an MPAES_RCS object with default parameters, variator and initializer. Finally, it shows the results by
plotting the pareto for the training set, another pareto evaluated for the testing set, and RB/DB for the classifier with
an accuracy that is the median among the values within the archive