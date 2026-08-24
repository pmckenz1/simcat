"""Deprecated compatibility module; use :mod:`simcat.database`."""

import warnings

warnings.warn(
    "simcat.Database is a deprecated module path; import Database from "
    "simcat.database or simcat instead.", DeprecationWarning, stacklevel=2,
)

from .database import Database

__all__ = ["Database"]
