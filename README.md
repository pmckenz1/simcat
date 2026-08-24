# simcat

`simcat` uses coalescent simulations and a neural-network classifier to infer a
directional introgression edge on a fixed species-tree topology from unlinked SNP
site-pattern frequencies.

This is an early research-software release. Version 0.1.0.dev0 supports Python
3.10 and the dependency versions pinned in `constraints/py310-tested.txt`. It
should be validated on
simulated data representative of a study before biological interpretation.

## Installation

Clone the repository and install it into a new Python 3.10 environment:

```bash
git clone https://github.com/pmckenz1/simcat.git
cd simcat
python -m pip install '.[hpc,ml]'
python -c "import simcat; print(simcat.__version__)"
```

The base installation is lightweight. Add `simulation`, `plot`, `hpc`, or `ml`
only for those workflows; `hpc` includes ipyparallel/Jupyter display support.
TensorFlow is loaded only when `simcat.BatchTrain` is first requested, not by a
normal `import simcat`. For the exact tested direct dependency versions, use:

```bash
python -m pip install -c constraints/py310-tested.txt '.[hpc,ml]'
```

The supported Python API, extras, schema fields, and compatibility rules are
defined in [`CONTRACTS.md`](CONTRACTS.md).

## Workflow

The published workflow has four stages: create simulation labels, simulate SNP
alignments, train/validate a classifier, and apply it to a new alignment.

The following is a deliberately small smoke example, not an adequate biological
training database:

```python
from pathlib import Path

import ipcoal
import simcat
import toytree

name = "quickstart"
workdir = Path("simcat-results")
tree = toytree.rtree.imbtree(ntips=5, treeheight=5e6)

# This creates new artifacts and refuses to replace an existing run. Choose a
# new name or explicitly pass force=True if replacement is really intended.
simcat.Database(
    name,
    workdir,
    tree,
    nrows=100,
    nsnps=500,
    Ne_min=50_000,
    Ne_max=150_000,
    admix_prop_min=0.1,
    admix_prop_max=0.5,
    admix_edge_min=0.1,
    admix_edge_max=0.9,
    exclude_sisters=True,
    node_slide_prop=0.5,
    seed=123,
)

simulator = simcat.Simulator(name, workdir)
simulator.run(auto=True)  # starts a temporary local ipyparallel cluster
print(simulator.status())

trainer = simcat.BatchTrain(
    input_name=name,
    output_name=name,
    directory=workdir,
    prop_training=0.9,
    exclude_sisters=True,
    exclude_magnitude=0,
    seed=123,
)
trainer.init_model()
trainer.train(batch_size=20, num_epochs=1)

# Example prediction from a simulated alignment. Alignment rows must be in the
# exact tree-tip order and values must be integer allele codes 0, 1, 2, or 3.
model = ipcoal.Model(
    tree=tree,
    admixture_edges=[(5, 3, 0.5, 0.25)],
    Ne=100_000,
    mut=1e-8,
)
model.sim_snps(500)
prediction = trainer.predict_from_alignment(model.seqs)
print(prediction)
```

Use substantially more simulations and SNPs for research. The manuscript uses
60,000 simulations for its larger examples. Simulation is normally the largest
compute cost; TensorFlow training may also need substantial CPU/GPU time and
memory. Start with a small benchmark using the actual tree and parameter ranges
before requesting cluster resources.

The executable walkthrough is in
[`notebooks/simcat_demo.ipynb`](notebooks/simcat_demo.ipynb). The Slurm pattern is
shown in [`notebooks/HPC_slurm_demo.ipynb`](notebooks/HPC_slurm_demo.ipynb).

## Artifacts and restart behavior

For a run named `quickstart`, simcat creates:

- `quickstart.labels.h5`: tree, sampled parameters, and row status;
- `quickstart.counts.db`: concurrent SQLite simulation results;
- `quickstart.counts.h5`: training copy synchronized from SQLite when
  `BatchTrain` is initialized;
- `quickstart.analysis.h5` and `quickstart.onehot_dict.csv`: split and category
  metadata; and
- `quickstart.model.h5`: saved Keras model; and
- `quickstart.model.h5.metadata.json`: schema, feature-order, category, seed,
  configuration, and package-version metadata for that model.

New database and model artifacts use schema version 1. Existing 0.0.6/0.0.7
artifacts can be inspected without mutation or migrated with
`python -m simcat.artifacts`; see the exact commands and backup guidance in
[`CONTRACTS.md`](CONTRACTS.md#legacy-migration).

Do not initialize `BatchTrain` while simulation jobs are still writing. Python
task failures release their own reserved rows. After a hard process or scheduler
interruption, first confirm no other worker is using the database, then run:

```python
simulator = simcat.Simulator("quickstart", "simcat-results")
print(simulator.status())
simulator.recover()
simulator.run(auto=True)
```

Multiple Slurm jobs may call `Simulator.run` against the same database; row
reservation is serialized and SNP arrays are committed through SQLite. Never use
`recover()` while any of those jobs is active. The HPC notebook contains a
parameterized template; replace its environment activation placeholder with the
site-specific command for the target cluster.

## Important limitations

- The input species-tree topology is fixed. simcat does not search tree or
  network space.
- The current model always selects an introgression-edge category; it has no
  validated "no introgression" class or stopping rule.
- Prediction currently accepts only an in-memory, two-dimensional integer SNP
  array. It does not yet map sample names from FASTA/VCF files or handle missing
  and ambiguous allele codes.
- Alignment rows must exactly match tree-tip order. Version 0.0.7 validates the
  shape and allele range but cannot validate biological sample identity.
- Softmax outputs are classifier scores and are not calibrated probabilities.
- Sister-edge inference, multiple samples per species, selection, and robustness
  beyond the simulated training domain are not established.
- Existing 0.0.6/0.0.7 model and database artifacts are treated as legacy
  schema 0. Back them up before running the in-place metadata migration. This
  release retains their scientific arrays and model format while adding explicit
  contracts and stricter validation.

## Citation

McKenzie, P. F., and D. A. R. Eaton. 2026. Detecting introgression from
phylogenetic invariant site patterns using machine learning. *Applications in
Plant Sciences* 14: e70061. <https://doi.org/10.1002/aps3.70061>

## License and support

simcat is distributed under the GPLv3 license; see [`LICENSE`](LICENSE).
Questions and reproducible bug reports may be submitted through the GitHub
repository or sent to Patrick McKenzie at `patrick.mckenzie@oregonstate.edu`.
