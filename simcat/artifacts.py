"""Versioned metadata readers and migrations for simcat artifacts."""

from __future__ import annotations

import argparse
import itertools
import json
from importlib import metadata as importlib_metadata
from math import comb
import os
from pathlib import Path
import shlex
import sqlite3
import tempfile
from typing import Any, Mapping, Optional, Sequence

import h5py


DATABASE_SCHEMA_VERSION = 1
MODEL_SCHEMA_VERSION = 1
FEATURE_SCHEMA_VERSION = 1
METADATA_ATTRIBUTE = "simcat_metadata_json"
SCHEMA_ATTRIBUTE = "simcat_schema_version"
FEATURE_NORMALIZATION = "per_quartet_max"


class ArtifactCompatibilityError(ValueError):
    """An artifact cannot satisfy the requested schema contract."""


_DATABASE_METADATA_FIELDS = {
    "artifact_type",
    "schema_version",
    "feature_schema_version",
    "package_versions",
    "tree_newick",
    "tip_order",
    "quartet_order",
    "edge_category_map",
    "feature_normalization",
    "seeds",
    "configuration",
}
_MODEL_METADATA_FIELDS = _DATABASE_METADATA_FIELDS.union({"input_shape"})


def _validate_metadata_document(
    document: Mapping[str, Any], artifact_type: str, required: set[str]
) -> dict[str, Any]:
    """Validate the fields needed to interpret a current-schema artifact."""
    missing = sorted(required.difference(document))
    if missing:
        raise ArtifactCompatibilityError(
            f"{artifact_type} metadata is missing required fields: {missing}."
        )
    if document.get("artifact_type") != artifact_type:
        raise ArtifactCompatibilityError(
            f"Expected {artifact_type} metadata, found "
            f"{document.get('artifact_type')!r}."
        )
    if document.get("feature_normalization") != FEATURE_NORMALIZATION:
        raise ArtifactCompatibilityError(
            "Unsupported feature normalization "
            f"{document.get('feature_normalization')!r}; expected "
            f"{FEATURE_NORMALIZATION!r}."
        )
    return dict(document)


def dependency_versions(names: Sequence[str] = ()) -> dict[str, Optional[str]]:
    """Return versions without importing optional dependency modules."""
    selected = ("simcat", "numpy", "h5py", "toytree", *names)
    versions: dict[str, Optional[str]] = {}
    for name in dict.fromkeys(selected):
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def encode_metadata(metadata: Mapping[str, Any]) -> str:
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"))


def database_metadata(
    *,
    tree_newick: str,
    tip_order: Sequence[str],
    quartet_order: Sequence[Sequence[int]],
    edge_category_map: Mapping[str, Sequence[int]],
    seeds: Mapping[str, Any],
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the canonical database schema-1 metadata document."""
    return {
        "artifact_type": "simcat-database",
        "schema_version": DATABASE_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "package_versions": dependency_versions(("ipcoal", "msprime")),
        "tree_newick": tree_newick,
        "tip_order": list(tip_order),
        "quartet_order": [list(quartet) for quartet in quartet_order],
        "edge_category_map": {
            str(key): [int(value[0]), int(value[1])]
            for key, value in edge_category_map.items()
        },
        "feature_normalization": FEATURE_NORMALIZATION,
        "seeds": dict(seeds),
        "configuration": dict(configuration),
    }


def write_hdf5_metadata(handle: h5py.File, metadata: Mapping[str, Any]) -> None:
    handle.attrs[SCHEMA_ATTRIBUTE] = int(metadata["schema_version"])
    handle.attrs[METADATA_ATTRIBUTE] = encode_metadata(metadata)


def _decode_hdf5_metadata(handle: h5py.File) -> Optional[dict[str, Any]]:
    encoded = handle.attrs.get(METADATA_ATTRIBUTE)
    if encoded is None:
        return None
    if isinstance(encoded, bytes):
        encoded = encoded.decode("utf-8")
    try:
        document = json.loads(str(encoded))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(
            "Artifact metadata is not valid JSON; the file may be corrupted."
        ) from exc
    if int(document.get("schema_version", -1)) != int(
        handle.attrs.get(SCHEMA_ATTRIBUTE, -1)
    ):
        raise ArtifactCompatibilityError(
            "Artifact schema attributes disagree; the file may be corrupted."
        )
    return document


def _tip_labels_from_newick(newick: str, ntips: int) -> list[str]:
    """Infer leaf labels from Newick while keeping legacy reading lightweight."""
    try:
        import toytree

        return [str(tip) for tip in toytree.tree(newick).get_tip_labels()]
    except Exception:
        return [str(index) for index in range(ntips)]


def infer_legacy_database_metadata(labels_path: Path) -> dict[str, Any]:
    """Read a 0.0.6/0.0.7 labels file under an explicit legacy contract."""
    with h5py.File(labels_path, "r") as labels:
        required = {
            "tree",
            "ntips",
            "nquarts",
            "nsnps",
        }
        missing = sorted(required.difference(labels.attrs.keys()))
        required_datasets = {"admixture", "finished_sims"}
        missing_datasets = sorted(required_datasets.difference(labels.keys()))
        if missing or missing_datasets:
            raise ArtifactCompatibilityError(
                "Unrecognized legacy database artifact; missing "
                f"attributes {missing} and datasets {missing_datasets}."
            )
        newick = str(labels.attrs["tree"])
        ntips = int(labels.attrs["ntips"])
        nquarts = int(labels.attrs["nquarts"])
        expected_nquarts = sum(1 for _ in itertools.combinations(range(ntips), 4))
        if nquarts != expected_nquarts:
            raise ArtifactCompatibilityError(
                f"Legacy nquarts={nquarts} does not match ntips={ntips}."
            )
        admixture = labels["admixture"][:, -1, :2]
        categories = sorted({(int(row[0]), int(row[1])) for row in admixture})
        seed = int(labels.attrs.get("seed", -1))
        nsnps = int(labels.attrs["nsnps"])
        release = "0.0.7" if "pending_counts_are_null" in labels.attrs else "0.0.6"
    tips = _tip_labels_from_newick(newick, ntips)
    return {
        "artifact_type": "simcat-database",
        "schema_version": 0,
        "legacy_release": release,
        "feature_schema_version": 0,
        "package_versions": {"simcat": release},
        "tree_newick": newick,
        "tip_order": tips,
        "quartet_order": [
            list(quartet) for quartet in itertools.combinations(range(ntips), 4)
        ],
        "edge_category_map": {
            str(index): list(edge) for index, edge in enumerate(categories)
        },
        "feature_normalization": FEATURE_NORMALIZATION,
        "seeds": {"master": None if seed < 0 else seed},
        "configuration": {
            "nrows": int(admixture.shape[0]),
            "nsnps": nsnps,
            "legacy_release": release,
        },
    }


def database_migration_command(
    labels_path: Path, counts_path: Optional[Path] = None,
    sqlite_path: Optional[Path] = None,
) -> str:
    command = [
        "python", "-m", "simcat.artifacts", "migrate-database",
        "--labels", str(labels_path),
    ]
    if counts_path is not None:
        command.extend(("--counts", str(counts_path)))
    if sqlite_path is not None:
        command.extend(("--sqlite", str(sqlite_path)))
    return " ".join(shlex.quote(part) for part in command)


def read_database_metadata(
    labels_path: Path | str,
    *,
    require_current: bool = False,
    counts_path: Path | str | None = None,
    sqlite_path: Path | str | None = None,
) -> dict[str, Any]:
    labels_path = Path(labels_path)
    with h5py.File(labels_path, "r") as labels:
        metadata = _decode_hdf5_metadata(labels)
    if metadata is None:
        metadata = infer_legacy_database_metadata(labels_path)
        if require_current:
            command = database_migration_command(
                labels_path,
                Path(counts_path) if counts_path is not None else None,
                Path(sqlite_path) if sqlite_path is not None else None,
            )
            raise ArtifactCompatibilityError(
                f"Legacy simcat {metadata['legacy_release']} database schema 0. "
                f"Migrate metadata with: {command}"
            )
    if int(metadata["schema_version"]) > DATABASE_SCHEMA_VERSION:
        raise ArtifactCompatibilityError(
            f"Database schema {metadata['schema_version']} is newer than this "
            f"simcat supports ({DATABASE_SCHEMA_VERSION})."
        )
    if int(metadata["schema_version"]) == DATABASE_SCHEMA_VERSION:
        metadata = _validate_metadata_document(
            metadata, "simcat-database", _DATABASE_METADATA_FIELDS
        )
    return metadata


def migrate_database(
    labels_path: Path | str,
    *,
    counts_path: Path | str | None = None,
    sqlite_path: Path | str | None = None,
) -> dict[str, Any]:
    """Add schema-1 metadata to a recognized 0.0.6/0.0.7 artifact set."""
    labels_path = Path(labels_path)
    legacy = read_database_metadata(labels_path)
    if int(legacy["schema_version"]) == DATABASE_SCHEMA_VERSION:
        migrated = legacy
    else:
        migrated = dict(legacy)
        migrated["schema_version"] = DATABASE_SCHEMA_VERSION
        migrated["feature_schema_version"] = FEATURE_SCHEMA_VERSION
        migrated["migrated_from"] = {
            "schema_version": 0,
            "release": legacy["legacy_release"],
        }
        migrated["package_versions"] = dependency_versions(("ipcoal", "msprime"))
        with h5py.File(labels_path, "r+") as labels:
            write_hdf5_metadata(labels, migrated)
    if counts_path is not None:
        with h5py.File(Path(counts_path), "r+") as counts:
            write_hdf5_metadata(counts, migrated)
    if sqlite_path is not None:
        connection = sqlite3.connect(Path(sqlite_path))
        try:
            connection.execute(
                "create table if not exists simcat_metadata "
                "(schema_version integer not null, document text not null)"
            )
            connection.execute("delete from simcat_metadata")
            connection.execute(
                "insert into simcat_metadata values (?, ?)",
                (DATABASE_SCHEMA_VERSION, encode_metadata(migrated)),
            )
            connection.commit()
        finally:
            connection.close()
    return migrated


def model_metadata_path(model_path: Path | str) -> Path:
    return Path(str(model_path) + ".metadata.json")


def write_model_metadata(model_path: Path | str, metadata: Mapping[str, Any]) -> Path:
    path = model_metadata_path(model_path)
    document = dict(metadata)
    document["artifact_type"] = "simcat-model"
    document["schema_version"] = MODEL_SCHEMA_VERSION
    _validate_metadata_document(document, "simcat-model", _MODEL_METADATA_FIELDS)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}-",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def model_migration_command(
    model_path: Path, analysis_path: Optional[Path] = None,
    categories_path: Optional[Path] = None,
) -> str:
    command = [
        "python", "-m", "simcat.artifacts", "migrate-model",
        "--model", str(model_path),
    ]
    if analysis_path is not None:
        command.extend(("--analysis", str(analysis_path)))
    if categories_path is not None:
        command.extend(("--categories", str(categories_path)))
    return " ".join(shlex.quote(part) for part in command)


def read_model_metadata(
    model_path: Path | str,
    *,
    require_current: bool = False,
    analysis_path: Path | str | None = None,
    categories_path: Path | str | None = None,
) -> dict[str, Any]:
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    path = model_metadata_path(model_path)
    if not path.exists():
        if require_current:
            raise ArtifactCompatibilityError(
                "Legacy simcat model has no schema sidecar. Migrate metadata with: "
                + model_migration_command(
                    model_path,
                    Path(analysis_path) if analysis_path is not None else None,
                    Path(categories_path) if categories_path is not None else None,
                )
            )
        return {
            "artifact_type": "simcat-model",
            "schema_version": 0,
            "legacy_release": "0.0.6/0.0.7",
        }
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(
            f"Model metadata sidecar is unreadable or malformed: {path}"
        ) from exc
    version = int(document.get("schema_version", -1))
    if version > MODEL_SCHEMA_VERSION:
        raise ArtifactCompatibilityError(
            f"Model schema {version} is newer than this simcat supports "
            f"({MODEL_SCHEMA_VERSION})."
        )
    if version == MODEL_SCHEMA_VERSION:
        return _validate_metadata_document(
            document, "simcat-model", _MODEL_METADATA_FIELDS
        )
    raise ArtifactCompatibilityError(
        f"Unsupported model schema {version}. Migrate metadata with: "
        + model_migration_command(
            model_path,
            Path(analysis_path) if analysis_path is not None else None,
            Path(categories_path) if categories_path is not None else None,
        )
    )


def migrate_model(
    model_path: Path | str,
    *,
    analysis_path: Path | str,
    categories_path: Path | str,
) -> dict[str, Any]:
    """Create a schema-1 sidecar for a legacy Keras/analysis/CSV artifact set."""
    import csv

    model_path = Path(model_path)
    analysis_path = Path(analysis_path)
    categories_path = Path(categories_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    with h5py.File(analysis_path, "r") as analysis:
        newick = str(analysis.attrs["newick"])
        nquarts = int(analysis.attrs["nquarts"])
        seed = int(analysis.attrs.get("seed", -1))
        input_shape = [int(value) for value in analysis.attrs["input_shape"]]
    ntips = next(
        (
            candidate
            for candidate in range(4, 10_000)
            if comb(candidate, 4) == nquarts
        ),
        None,
    )
    if ntips is None:
        raise ArtifactCompatibilityError(
            f"Legacy quartet count {nquarts} cannot be mapped to a tip count."
        )
    tip_order = _tip_labels_from_newick(newick, ntips)
    with categories_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    if len(rows) < 3:
        raise ArtifactCompatibilityError("Legacy category CSV is malformed.")
    category_map = {
        str(index): value for index, value in enumerate(rows[2][1:])
    }
    metadata = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_normalization": FEATURE_NORMALIZATION,
        "tree_newick": newick,
        "tip_order": tip_order,
        "quartet_order": [
            list(quartet) for quartet in itertools.combinations(range(ntips), 4)
        ],
        "quartet_count": nquarts,
        "input_shape": input_shape,
        "edge_category_map": category_map,
        "seeds": {"training": None if seed < 0 else seed},
        "package_versions": dependency_versions(("tensorflow", "pandas")),
        "configuration": {"migrated_from": "0.0.6/0.0.7"},
    }
    write_model_metadata(model_path, metadata)
    return metadata | {"artifact_type": "simcat-model", "schema_version": 1}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    database = commands.add_parser("migrate-database")
    database.add_argument("--labels", required=True, type=Path)
    database.add_argument("--counts", type=Path)
    database.add_argument("--sqlite", type=Path)
    model = commands.add_parser("migrate-model")
    model.add_argument("--model", required=True, type=Path)
    model.add_argument("--analysis", required=True, type=Path)
    model.add_argument("--categories", required=True, type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "migrate-database":
        migrate_database(args.labels, counts_path=args.counts, sqlite_path=args.sqlite)
    else:
        migrate_model(
            args.model, analysis_path=args.analysis, categories_path=args.categories
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
