"""Stable public API for simcat.

Optional simulation, plotting, HPC, and TensorFlow modules are loaded only when
their public object is requested.
"""

from importlib import import_module
from types import ModuleType
import sys

from .config import (
    DatabaseConfig,
    ParameterRanges,
    RNGConfig,
    StorageConfig,
    SubstitutionModelConfig,
    TrainingConfig,
    TreeConfig,
)

__version__ = "0.1.0.dev0"
__authors__ = "Patrick McKenzie and Deren Eaton"

__all__ = [
    "BatchTrain", "Database", "DatabaseConfig", "ParameterRanges",
    "RNGConfig", "Simulator", "StorageConfig", "SubstitutionModelConfig",
    "TrainingConfig", "TreeConfig", "plot",
]

_LAZY_EXPORTS = {
    "Database": (".database", "Database"),
    "Simulator": (".simulator", "Simulator"),
    "BatchTrain": (".training", "BatchTrain"),
    "plot": (".plot", None),
}


def __getattr__(name):
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


class _SimcatModule(ModuleType):
    """Keep class exports stable after importing deprecated module paths.

    Python normally assigns ``simcat.Database`` to the compatibility module
    object after ``import simcat.Database``. Intercepting that one ambiguous
    package lookup preserves the documented top-level class export while the
    deprecated module itself remains importable.
    """

    def __getattribute__(self, name):
        value = super().__getattribute__(name)
        if name in {"Database", "Simulator", "BatchTrain"} and isinstance(
            value, ModuleType
        ):
            exports = super().__getattribute__("_LAZY_EXPORTS")
            module_name, attribute = exports[name]
            value = getattr(import_module(module_name, __name__), attribute)
            setattr(self, name, value)
        return value


sys.modules[__name__].__class__ = _SimcatModule
