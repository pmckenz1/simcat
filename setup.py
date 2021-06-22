#!/usr/bin/env python

from setuptools import setup, find_packages
import re

# parse version from init.py
with open("simcat/__init__.py") as init:
    CUR_VERSION = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]",
        init.read(),
        re.M,
    ).group(1)

# run setup script
setup(
    name="simcat",
    version=CUR_VERSION,
    url="https://github.com/pmckenz1/simcat",
    author="Patrick McKenzie and Deren Eaton",
    author_email="de2356@columbia.edu",
    description="simulation and machine learning algorithms for admixture inference",
    long_description=open('README.md').read(),
    packages=find_packages(),
    install_requires=[
        "future",
        "ipcoal",
        "h5py",
        "ipyparallel",
        "ipywidgets",
        "fasteners",
        # "numba",
        # "numpy",
        # "scipy",
        # "toytree",
        # "msprime",
        # "ipython",
    ],
    keywords="invariants coalescent simulation genomics introgression",
    entry_points={},
    data_files=[],
    license='GPLv3',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Framework :: Jupyter'        
    ],
)
