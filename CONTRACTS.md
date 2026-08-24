# Public API and artifact contracts

This document defines the compatibility boundary for the Phase 2 development
release, `simcat` 0.1.0.dev0. Scientific and storage changes after this point
must either preserve these contracts or increment the relevant schema version
and provide a migration.

## Public Python API

The supported implementation modules are lower case:

- `simcat.database.Database`
- `simcat.simulator.Simulator`
- `simcat.training.BatchTrain`
- `simcat.config.DatabaseConfig` and `TrainingConfig`, together with the nested
  tree, parameter-range, substitution-model, RNG, and storage contracts
- `simcat.artifacts` readers and migrations

`Database`, `Simulator`, and `BatchTrain` are also lazy top-level exports. The
old capitalized module paths remain importable for 0.1.x and emit
`DeprecationWarning`; new code should not use them. Optional modules are loaded
only when requested, so `import simcat` needs neither the simulation, plotting,
HPC, nor ML extras.

Configuration objects are frozen dataclasses. `DatabaseConfig` and
`TrainingConfig` provide `to_dict()` and `from_dict()` methods whose results can
be encoded as JSON. Paths are serialized as strings, collection fields use JSON
arrays, and restoration revalidates every field.

## Installation profiles

The base installation contains feature construction, HDF5 artifact inspection,
configuration, and database label generation. Optional extras are:

| Extra | Purpose |
| --- | --- |
| `simulation` | ipcoal/msprime simulation and local locking |
| `plot` | toyplot visualizations |
| `hpc` | simulation plus ipyparallel and notebook progress displays |
| `ml` | pandas and TensorFlow training/prediction |
| `test` | lightweight build and contract tests |
| `dev` | all dependencies used by the full test suite |

`constraints/py310-tested.txt` pins the direct dependencies from the tested
Python 3.10 environment. Install an extra through that file to reproduce the
validated direct dependency set; transitive dependency resolution remains the
responsibility of pip and should be captured with `pip freeze` for archived
analyses.

## Database schema 1

A schema-1 database stores the same scientific arrays as 0.0.7. The labels and
counts HDF5 files have `simcat_schema_version=1` and a canonical JSON document
in `simcat_metadata_json`. The SQLite database stores that version and document
in a one-row `simcat_metadata` table.

The JSON document records:

- artifact and feature schema versions;
- simcat and scientific dependency versions;
- species-tree Newick and explicit tip order;
- quartet index order and category-to-directional-edge mapping;
- the published `per_quartet_max` feature normalization;
- master and per-row seed semantics; and
- the complete database configuration, including storage identity, simulation
  ranges, substitution model, and existing admixture edges.

`read_database_metadata()` reads schema 1 and recognized 0.0.6/0.0.7 labels
files. Unknown, corrupt, or newer schemas raise `ArtifactCompatibilityError`.
Set `require_current=True` when a workflow cannot safely operate on inferred
legacy metadata.

## Model schema 1

The Keras HDF5 model remains unchanged in Phase 2. A schema-1 model adds an
atomic `<name>.model.h5.metadata.json` sidecar containing the tree, tip and
quartet order, category map, input shape, normalization, split/model seeds,
training configuration, and relevant package versions. Analysis HDF5 files
also carry machine-readable schema metadata.

Model files without a sidecar are treated as legacy schema 0. Call
`read_model_metadata(..., require_current=True)` when inferred metadata is not
sufficient.

## Legacy migration

Readers do not mutate legacy artifacts. To add schema-1 metadata in place,
first back up the complete artifact set, then run:

```bash
python -m simcat.artifacts migrate-database \
  --labels run.labels.h5 \
  --counts run.counts.h5 \
  --sqlite run.counts.db

python -m simcat.artifacts migrate-model \
  --model run.model.h5 \
  --analysis run.analysis.h5 \
  --categories run.onehot_dict.csv
```

The compatibility exception includes the exact applicable command when current
metadata is required. Database migration preserves all existing arrays and
SQLite results; model migration preserves the Keras file and writes only its
JSON sidecar. A migrated artifact records its source release/schema.

## Versioning rules

- Database, model, and feature schemas are versioned independently.
- Adding optional metadata without changing interpretation is backward
  compatible. Removing required fields or changing their meaning is not.
- A changed tip/quartet/category order or normalization is a feature-schema
  change even if array shapes stay equal.
- Readers reject future schema versions rather than guessing.
- The Phase 1 hashes and legacy fixtures under `tests/fixtures` are the golden
  compatibility baseline for label generation and feature ordering.

