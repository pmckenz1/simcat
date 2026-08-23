import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )


def test_main_notebook_matches_training_api_and_has_no_stale_output():
    path = ROOT / "notebooks" / "simcat_demo.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = _source(path)
    assert "mod.init_model()" in source
    assert "workers=" not in source
    assert all(not cell.get("outputs") for cell in notebook["cells"])


def test_hpc_notebook_has_no_developer_specific_paths():
    path = ROOT / "notebooks" / "HPC_slurm_demo.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = _source(path)
    assert "/n/home09" not in source
    assert "/moto/" not in source
    assert "database_dir = os.path.abspath" in source
    assert "placeholder" in source
    assert all(not cell.get("outputs") for cell in notebook["cells"])
