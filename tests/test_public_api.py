import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_import_does_not_load_tensorflow():
    code = (
        "import sys, simcat; "
        "assert simcat.__version__ == '0.0.7'; "
        "assert 'tensorflow' not in sys.modules; "
        "assert set(simcat.__all__) == "
        "{'BatchTrain', 'Database', 'Simulator', 'plot'}"
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


def test_setup_metadata_smoke():
    result = subprocess.run(
        [sys.executable, "setup.py", "--name", "--version"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.stdout.splitlines() == ["simcat", "0.0.7"]


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
    wheels = list(tmp_path.glob("simcat-0.0.7-*.whl"))
    assert len(wheels) == 1
