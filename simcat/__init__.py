"""Public API for simcat.

TensorFlow is intentionally imported only when ``BatchTrain`` is requested.
"""

from .Database import Database
from .Simulator import Simulator
from . import plot

__all__ = ["BatchTrain", "Database", "Simulator", "plot"]


def __getattr__(name):
    if name == "BatchTrain":
        # Import BatchTrain (and thus TensorFlow) only when accessed. Assigning
        # the class here also replaces the package attribute that Python creates
        # temporarily for the imported ``simcat.BatchTrain`` submodule.
        from .BatchTrain import BatchTrain as batch_train_class

        globals()[name] = batch_train_class
        return batch_train_class
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "0.0.7"
__authors__ = "Patrick McKenzie and Deren Eaton"
