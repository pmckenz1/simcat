import hashlib
import json
from pathlib import Path
import sqlite3

import h5py
import numpy as np
import pytest
import toytree

from simcat import Database
from simcat.artifacts import read_database_metadata


LABEL_DATASETS = (
    "node_heights",
    "node_Nes",
    "slide_seeds",
    "treeheight",
    "admixture",
)
GOLDEN_PATH = Path(__file__).parent / "fixtures" / "phase1_database_golden.json"


def _make_database(workdir, tree, **kwargs):
    return Database(
        "test",
        workdir,
        tree,
        nrows=5,
        nsnps=12,
        seed=123,
        quiet=True,
        **kwargs,
    )


def test_seed_controls_labels_for_tree_object_and_newick(tmp_path):
    tree = toytree.rtree.imbtree(ntips=4, treeheight=1e6)
    first = _make_database(tmp_path / "first", tree)
    second = _make_database(tmp_path / "second", tree.write())
    with h5py.File(first.labels, "r") as left, h5py.File(
        second.labels, "r"
    ) as right:
        for key in LABEL_DATASETS:
            np.testing.assert_array_equal(left[key][:], right[key][:])
        assert left.attrs["seed"] == 123

    with sqlite3.connect(first.sqldb) as connection:
        placeholders = connection.execute(
            "select count(*) from counts where arr is null"
        ).fetchone()[0]
    assert placeholders == 5


def test_phase1_label_generation_golden_compatibility(tmp_path):
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    tree = toytree.rtree.imbtree(ntips=4, treeheight=1e6)
    database = Database(
        "golden",
        tmp_path,
        tree,
        nrows=golden["nrows"],
        nsnps=golden["nsnps"],
        seed=golden["seed"],
        quiet=True,
    )
    with h5py.File(database.labels, "r") as labels:
        observed = {
            name: hashlib.sha256(labels[name][:].tobytes()).hexdigest()
            for name in golden["sha256"]
        }
    assert observed == golden["sha256"]


def test_new_database_records_the_schema_contract(tmp_path):
    tree = toytree.rtree.imbtree(ntips=5, treeheight=1e6)
    database = _make_database(tmp_path, tree)
    metadata = read_database_metadata(database.labels, require_current=True)
    assert metadata["artifact_type"] == "simcat-database"
    assert metadata["schema_version"] == 1
    assert metadata["feature_schema_version"] == 1
    assert metadata["feature_normalization"] == "per_quartet_max"
    assert metadata["tip_order"] == tree.get_tip_labels()
    assert metadata["quartet_order"] == [
        [0, 1, 2, 3],
        [0, 1, 2, 4],
        [0, 1, 3, 4],
        [0, 2, 3, 4],
        [1, 2, 3, 4],
    ]
    assert metadata["seeds"]["master"] == 123
    assert metadata["configuration"]["nrows"] == 5


def test_existing_artifacts_are_protected_and_force_recreates_all(tmp_path):
    tree = toytree.rtree.imbtree(ntips=4, treeheight=1e6)
    database = _make_database(tmp_path, tree)
    paths = [Path(database.labels), Path(database.counts), Path(database.sqldb)]
    with h5py.File(database.labels, "r+") as labelsfile:
        labelsfile.attrs["test_sentinel"] = True
    with h5py.File(database.counts, "r+") as countsfile:
        countsfile.attrs["test_sentinel"] = True
    with sqlite3.connect(database.sqldb) as connection:
        connection.execute("create table test_sentinel (value integer)")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _make_database(tmp_path, tree)
    assert all(path.exists() for path in paths)

    replacement = _make_database(tmp_path, tree, force=True)
    replaced_paths = [
        Path(replacement.labels),
        Path(replacement.counts),
        Path(replacement.sqldb),
    ]
    assert all(path.exists() for path in replaced_paths)
    with h5py.File(replacement.labels, "r") as labelsfile:
        assert "test_sentinel" not in labelsfile.attrs
    with h5py.File(replacement.counts, "r") as countsfile:
        assert "test_sentinel" not in countsfile.attrs
    with sqlite3.connect(replacement.sqldb) as connection:
        tables = connection.execute(
            "select name from sqlite_master where type='table'"
        ).fetchall()
    assert ("test_sentinel",) not in tables


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"nrows": 0}, "nrows"),
        ({"nsnps": 0}, "nsnps"),
        ({"admix_prop_min": 0.8, "admix_prop_max": 0.2}, "proportions"),
        ({"node_slide_prop": 1.1}, "node_slide_prop"),
        ({"rate_vector": np.ones(6)}, "both be supplied"),
        (
            {
                "rate_vector": np.ones(6),
                "pi_vector": np.array([0.2, 0.2, 0.2, 0.2]),
            },
            "sum to one",
        ),
    ],
)
def test_invalid_parameters_fail_before_artifacts_are_created(
    tmp_path, kwargs, message
):
    tree = toytree.rtree.imbtree(ntips=4, treeheight=1e6)
    defaults = {"nrows": 5, "nsnps": 12}
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=message):
        Database("test", tmp_path, tree, quiet=True, **defaults)
    assert not list(tmp_path.iterdir())
