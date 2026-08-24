import numpy as np
import pytest
import toytree

pytest.importorskip("fasteners")
pytest.importorskip("ipcoal")
pytest.importorskip("msprime")

from simcat import Database, Simulator
from simcat.utils import SimcatError


pytestmark = pytest.mark.simulation


def _database(tmp_path, nrows=5):
    tree = toytree.rtree.imbtree(ntips=4, treeheight=1e6)
    Database(
        "test",
        tmp_path,
        tree,
        nrows=nrows,
        nsnps=10,
        seed=5,
        quiet=True,
    )
    return Simulator("test", tmp_path, quiet=True)


def test_reservations_are_bounded_conditional_and_recoverable(tmp_path):
    simulator = _database(tmp_path)
    reserved = simulator._reserve_simulations(100)
    np.testing.assert_array_equal(reserved, np.arange(5))
    assert simulator.status() == {
        "total": 5,
        "pending": 0,
        "complete": 0,
        "reserved": 5,
    }
    assert simulator._reserve_simulations(1).size == 0

    simulator._set_simulation_status(
        reserved[:2], simulator.COMPLETE, only_if=simulator.RESERVED
    )
    assert simulator.recover() == 3
    assert simulator.status() == {
        "total": 5,
        "pending": 3,
        "complete": 2,
        "reserved": 0,
    }


class _FailedJob:
    def ready(self):
        return True

    def successful(self):
        return False

    def get(self):
        raise RuntimeError("expected worker failure")


class _FailedView:
    def apply(self, function, *args):
        return _FailedJob()


class _FailedClient:
    def __len__(self):
        return 1

    def load_balanced_view(self):
        return _FailedView()


def test_python_worker_failure_releases_reserved_rows(tmp_path):
    simulator = _database(tmp_path)
    with pytest.raises(SimcatError, match="expected worker failure"):
        simulator._run(3, _FailedClient())
    assert simulator.status()["pending"] == 5
    assert simulator.status()["reserved"] == 0
