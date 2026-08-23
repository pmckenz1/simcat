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
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10,<3.11",
    install_requires=[
        "fasteners>=0.17,<1",
        "future>=0.18,<1",
        "h5py>=3.10,<4",
        "ipcoal>=0.4.dev6,<0.5",
        "ipython>=8.14,<9",
        "ipyparallel>=9,<10",
        "ipywidgets>=8,<9",
        "msprime>=1.2,<2",
        "numba>=0.57,<1",
        "numpy>=1.24,<2",
        "pandas>=2,<3",
        "tensorflow>=2.17,<2.18",
        "toyplot>=1,<2",
        "toytree>=3.0.4,<4",
    ],
    extras_require={"test": ["pytest>=7,<9"]},
    keywords="invariants coalescent simulation genomics introgression",
    entry_points={},
    data_files=[],
    license='GPLv3',
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Framework :: Jupyter",
    ],
)
