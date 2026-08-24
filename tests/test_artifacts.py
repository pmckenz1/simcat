import json
from pathlib import Path
import sqlite3

import h5py
import numpy as np
import pytest

from simcat.artifacts import (
    ArtifactCompatibilityError,
    DATABASE_SCHEMA_VERSION,
    METADATA_ATTRIBUTE,
    SCHEMA_ATTRIBUTE,
    migrate_database,
    migrate_model,
    read_database_metadata,
    read_model_metadata,
)


LEGACY_TREE = "((a:1,b:1):1,(c:1,d:1):1);"


def _legacy_database(path: Path, release: str) -> None:
    with h5py.File(path, "w") as labels:
        labels.attrs["tree"] = LEGACY_TREE
        labels.attrs["ntips"] = 4
        labels.attrs["nquarts"] = 1
        labels.attrs["nsnps"] = 12
        labels.attrs["seed"] = 123
        if release == "0.0.7":
            labels.attrs["pending_counts_are_null"] = True
        labels.create_dataset(
            "admixture",
            data=np.array([[[0, 2, 0.5, 0.2]], [[2, 0, 0.5, 0.3]]]),
        )
        labels.create_dataset("finished_sims", data=np.array([1, 0]))


@pytest.mark.parametrize("release", ["0.0.6", "0.0.7"])
def test_golden_legacy_database_reader_and_exact_migration_command(
    tmp_path, release
):
    labels_path = tmp_path / f"legacy-{release}.labels.h5"
    _legacy_database(labels_path, release)

    metadata = read_database_metadata(labels_path)
    assert metadata["schema_version"] == 0
    assert metadata["legacy_release"] == release
    assert metadata["tip_order"] == ["a", "b", "c", "d"]
    assert metadata["quartet_order"] == [[0, 1, 2, 3]]
    with pytest.raises(ArtifactCompatibilityError) as error:
        read_database_metadata(labels_path, require_current=True)
    assert str(error.value).endswith(
        f"python -m simcat.artifacts migrate-database --labels {labels_path}"
    )


def test_legacy_database_migration_updates_the_artifact_set(tmp_path):
    labels_path = tmp_path / "legacy.labels.h5"
    counts_path = tmp_path / "legacy.counts.h5"
    sqlite_path = tmp_path / "legacy.counts.db"
    _legacy_database(labels_path, "0.0.7")
    with h5py.File(counts_path, "w") as counts:
        counts.create_dataset("counts", shape=(2, 4, 12), dtype=np.int8)
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute("create table counts (id integer primary key, arr blob)")

    # A labels-only first pass models interruption before the other stores.
    migrated = migrate_database(labels_path)
    assert read_database_metadata(labels_path, require_current=True) == migrated
    migrated_retry = migrate_database(
        labels_path, counts_path=counts_path, sqlite_path=sqlite_path
    )
    assert migrated_retry == migrated
    assert migrated["schema_version"] == DATABASE_SCHEMA_VERSION
    assert migrated["migrated_from"] == {
        "schema_version": 0,
        "release": "0.0.7",
    }
    assert read_database_metadata(labels_path, require_current=True) == migrated
    with h5py.File(counts_path, "r") as counts:
        assert counts.attrs[SCHEMA_ATTRIBUTE] == DATABASE_SCHEMA_VERSION
        assert json.loads(counts.attrs[METADATA_ATTRIBUTE]) == migrated
    with sqlite3.connect(sqlite_path) as connection:
        stored = connection.execute(
            "select schema_version, document from simcat_metadata"
        ).fetchone()
    assert stored[0] == DATABASE_SCHEMA_VERSION
    assert json.loads(stored[1]) == migrated


def test_unknown_and_future_database_schemas_are_rejected(tmp_path):
    unknown = tmp_path / "unknown.h5"
    with h5py.File(unknown, "w") as artifact:
        artifact.attrs["tree"] = LEGACY_TREE
    with pytest.raises(ArtifactCompatibilityError, match="Unrecognized legacy"):
        read_database_metadata(unknown)

    future = tmp_path / "future.h5"
    with h5py.File(future, "w") as artifact:
        artifact.attrs[SCHEMA_ATTRIBUTE] = 999
        artifact.attrs[METADATA_ATTRIBUTE] = json.dumps({"schema_version": 999})
    with pytest.raises(ArtifactCompatibilityError, match="newer"):
        read_database_metadata(future)


def test_legacy_model_reader_and_migration(tmp_path):
    model_path = tmp_path / "legacy.model.h5"
    analysis_path = tmp_path / "legacy.analysis.h5"
    categories_path = tmp_path / "legacy.onehot_dict.csv"
    model_path.write_bytes(b"legacy keras placeholder")
    with h5py.File(analysis_path, "w") as analysis:
        analysis.attrs["newick"] = LEGACY_TREE
        analysis.attrs["nquarts"] = 1
        analysis.attrs["input_shape"] = (1, 256)
        analysis.attrs["seed"] = 17
    categories_path.write_text(
        ',0,1\n0,0,1\n1,"0,2","2,0"\n', encoding="utf-8"
    )

    assert read_model_metadata(model_path)["schema_version"] == 0
    with pytest.raises(ArtifactCompatibilityError) as error:
        read_model_metadata(
            model_path,
            require_current=True,
            analysis_path=analysis_path,
            categories_path=categories_path,
        )
    assert "migrate-model" in str(error.value)
    assert f"--analysis {analysis_path}" in str(error.value)

    migrated = migrate_model(
        model_path,
        analysis_path=analysis_path,
        categories_path=categories_path,
    )
    assert migrated["schema_version"] == 1
    assert migrated["tip_order"] == ["a", "b", "c", "d"]
    assert migrated["quartet_order"] == [[0, 1, 2, 3]]
    assert read_model_metadata(model_path, require_current=True) == migrated
