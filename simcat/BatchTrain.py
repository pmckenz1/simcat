"""Deprecated compatibility module; use :mod:`simcat.training`."""

import warnings

warnings.warn(
    "simcat.BatchTrain is a deprecated module path; import BatchTrain from "
    "simcat.training or simcat instead.", DeprecationWarning, stacklevel=2,
)

from .training import BatchTrain

__all__ = ["BatchTrain"]
