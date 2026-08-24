import pytest

pytest.importorskip("ipyparallel")
pytest.importorskip("ipywidgets")
pytest.importorskip("IPython")

from simcat.parallel import Parallel
from simcat.utils import SimcatError


pytestmark = pytest.mark.hpc


class _Tool:
    ipcluster = {
        "profile": "default",
        "timeout": 0,
        "cluster_id": "test",
        "cores": 1,
        "engines": "Local",
        "pids": {},
    }


class _NoEngineClient:
    def __len__(self):
        return 0


def test_client_is_retained_for_cleanup_when_no_engine_registers(monkeypatch):
    client = _NoEngineClient()
    monkeypatch.setattr(
        "simcat.parallel.ipp.Client",
        lambda **kwargs: client,
    )
    pool = Parallel(_Tool(), quiet=True, auto=True)
    with pytest.raises(SimcatError, match="No ipyparallel engines"):
        pool.wait_for_connection()
    assert pool.ipyclient is client
