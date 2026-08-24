"""Deprecated compatibility module; use :mod:`simcat.simulator`."""

import warnings

warnings.warn(
    "simcat.Simulator is a deprecated module path; import Simulator from "
    "simcat.simulator or simcat instead.", DeprecationWarning, stacklevel=2,
)

from .simulator import IPCoalWrapper, Simulator

__all__ = ["IPCoalWrapper", "Simulator"]
