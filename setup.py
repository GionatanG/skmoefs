import logging
import os

try:
    import setuptools
    from setuptools import find_packages, setup
except ImportError:
    raise ImportError(
        "'setuptools' is required but not installed. To install it, "
        "follow the instructions at "
        "https://pip.pypa.io/en/stable/installing/#installing-with-get-pip-py")


setup(
    name = "hpsklearn",
    version = '0.0.3',
    packages = find_packages(),
    scripts = [],
    url = 'http://github.com/GionatanG/skmoefs/',
    download_url = 'https://github.com/GionatanG/skmoefs/archive/master.zip',
    author = ' ',
    author_email = ' ',
    description = ' ',
    long_description = open('README.md').read(),
    keywords = ['', '', ''],
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Environment :: Console',
        'License :: OSI Approved :: BSD License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
    ],
    platforms = ['Linux', 'OS-X', 'Windows'],
    license = 'BSD',
    install_requires = [
        'scipy',
        'numba',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'platypus-opt' ,
    ]
    extras_require = {
        'xgboost':  ['xgboost==0.6a2'],
        'lightgbm': ['lightgbm==2.3.1']
    }
)
