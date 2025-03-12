#!/usr/bin/env

# imports

from .Database import Database    # BUILDS THE DATABASE OF LABELS
from .Analysis import Analysis          # POST-SIM ANALYSIS
from .Simulator import Simulator
from . import plot
#from .BatchTrain import BatchTrain            #
def __getattr__(name):
    if name == "BatchTrain":
        # Import BatchTrain (and thus TensorFlow) only when accessed
        from .BatchTrain import BatchTrain
        return BatchTrain
    raise AttributeError(f"module {__name__} has no attribute {name}")



# dunders
__version__ = "0.0.6"
__authors__ = "Patrick McKenzie and Deren Eaton"
