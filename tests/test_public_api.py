import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_import_does_not_load_tensorflow():
    code = (
        "import sys, simcat; "
        "assert simcat.__version__ == '0.1.0.dev0'; "
        "assert not ({'tensorflow', 'ipcoal', 'msprime', 'fasteners', "
        "'toyplot'} & set(sys.modules)); "
        "assert set(simcat.__all__) == "
        "{'BatchTrain', 'Database', 'DatabaseConfig', 'ParameterRanges', "
        "'RNGConfig', 'Simulator', 'StorageConfig', "
        "'SubstitutionModelConfig', 'TrainingConfig', 'TreeConfig', 'plot'}"
    )
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-simcat-tests")
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        check=True,
        timeout=120,
    )


def test_pyproject_is_the_metadata_source():
    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "simcat"' in metadata
    assert 'version = "0.1.0.dev0"' in metadata
    assert not (ROOT / "setup.py").exists()
    for extra in ("simulation", "plot", "hpc", "ml", "test", "dev"):
        assert f"{extra} = [" in metadata


def test_lowercase_database_api_and_deprecated_module_shim():
    from simcat import Database
    from simcat.database import Database as LowercaseDatabase

    assert Database is LowercaseDatabase
    with pytest.deprecated_call(match="deprecated module path"):
        from simcat.Database import Database as LegacyDatabase
    assert LegacyDatabase is LowercaseDatabase
    import simcat

    assert simcat.Database is LowercaseDatabase


def test_wheel_build_smoke(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(tmp_path),
            ".",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    wheels = list(tmp_path.glob("simcat-0.1.0.dev0-*.whl"))
    assert len(wheels) == 1
