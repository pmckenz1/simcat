import importlib

import pytest

import simcat


@pytest.mark.simulation
def test_simulator_capitalized_module_is_a_deprecation_shim():
    pytest.importorskip("fasteners")
    pytest.importorskip("ipcoal")
    pytest.importorskip("msprime")
    current = importlib.import_module("simcat.simulator")
    with pytest.deprecated_call(match="deprecated module path"):
        legacy = importlib.import_module("simcat.Simulator")
    assert legacy.Simulator is current.Simulator
    assert legacy.IPCoalWrapper is current.IPCoalWrapper
    assert simcat.Simulator is current.Simulator


@pytest.mark.ml
def test_training_capitalized_module_is_a_deprecation_shim():
    pytest.importorskip("tensorflow")
    pytest.importorskip("pandas")
    current = importlib.import_module("simcat.training")
    with pytest.deprecated_call(match="deprecated module path"):
        legacy = importlib.import_module("simcat.BatchTrain")
    assert legacy.BatchTrain is current.BatchTrain
    assert simcat.BatchTrain is current.BatchTrain
