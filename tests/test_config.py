import json

import pytest
import toytree

from simcat import (
    DatabaseConfig,
    ParameterRanges,
    RNGConfig,
    StorageConfig,
    SubstitutionModelConfig,
    TrainingConfig,
    TreeConfig,
)
from simcat.database import Database


LEGACY_TREE = "((a:1,b:1):1,(c:1,d:1):1);"


def test_database_config_round_trip_and_constructor(tmp_path):
    tree = toytree.rtree.imbtree(ntips=4, treeheight=1e6)
    config = DatabaseConfig(
        tree=TreeConfig.from_tree(tree),
        storage=StorageConfig("configured", tmp_path),
        parameters=ParameterRanges(Ne_min=20_000, Ne_max=30_000),
        rng=RNGConfig(42),
        nrows=3,
        nsnps=8,
        quiet=True,
    )
    restored = DatabaseConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert restored == config

    database = Database.from_config(restored)
    assert database.nrows == 3
    assert database.nsnps == 8
    assert database.seed == 42


def test_training_config_round_trip(tmp_path):
    config = TrainingConfig(
        input_name="source",
        output_name="model",
        directory=tmp_path,
        batch_size=4,
        num_epochs=2,
        seed=9,
    )
    restored = TrainingConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert restored == config


@pytest.mark.parametrize(
    "factory, message",
    [
        (lambda: RNGConfig(-1), "seed"),
        (lambda: StorageConfig("../escape"), "path components"),
        (lambda: ParameterRanges(Ne_min=0), "Ne values"),
        (
            lambda: TreeConfig(LEGACY_TREE, ("d", "c", "b", "a")),
            "tip_order",
        ),
        (
            lambda: SubstitutionModelConfig(
                "GTR", rate_vector=(1, 1, 1, 1, 1, 1), pi_vector=(1, 1, 1, 1)
            ),
            "sum to one",
        ),
        (
            lambda: TrainingConfig("in", "out", ".", feature_normalization="sum"),
            "per_quartet_max",
        ),
    ],
)
def test_invalid_contracts_fail_early(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()
