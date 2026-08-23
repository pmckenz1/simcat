import h5py
import numpy as np
import pytest
import toytree

import simcat
from simcat.utils import SimcatError


class _ImmediateJob:
    def __init__(self, function, *args):
        try:
            self.value = function(*args)
            self.error = None
        except Exception as exc:  # pragma: no cover - exposed by get()
            self.value = None
            self.error = exc

    def ready(self):
        return True

    def successful(self):
        return self.error is None

    def get(self):
        if self.error is not None:
            raise self.error
        return self.value


class _ImmediateView:
    def apply(self, function, *args):
        return _ImmediateJob(function, *args)


class _ImmediateClient:
    def __len__(self):
        return 1

    def load_balanced_view(self):
        return _ImmediateView()


@pytest.mark.integration
def test_tiny_database_simulation_training_and_prediction(tmp_path):
    tree = toytree.rtree.imbtree(ntips=5, treeheight=1e6)
    simcat.Database(
        "tiny",
        tmp_path,
        tree,
        nrows=24,
        nsnps=40,
        seed=17,
        exclude_sisters=True,
        quiet=True,
    )
    simulator = simcat.Simulator("tiny", tmp_path, quiet=True)
    assert simulator._run(None, _ImmediateClient()) == 24
    assert simulator.status()["complete"] == 24

    batch_train_class = simcat.BatchTrain
    assert simcat.BatchTrain is batch_train_class
    trainer = batch_train_class(
        input_name="tiny",
        output_name="tiny",
        directory=tmp_path,
        prop_training=0.75,
        exclude_sisters=True,
        exclude_magnitude=0,
        seed=17,
    )
    with h5py.File(trainer.counts_filepath, "r") as countsfile:
        assert countsfile.attrs["sqlite_synchronized"]
        alignment = countsfile["counts"][0][:]
        assert alignment.sum() > 0

    with pytest.raises(SimcatError, match="No neural network"):
        trainer.predict_from_alignment(alignment)
    trainer.init_model(save=True)
    history = trainer.train(batch_size=8, num_epochs=1)
    assert len(history.history["loss"]) == 1

    prediction = trainer.predict_from_alignment(alignment).to_numpy()[0]
    assert np.isfinite(prediction).all()
    assert np.isclose(prediction.sum(), 1.0, atol=1e-5)
    with pytest.raises(ValueError, match="allele codes"):
        trainer.predict_from_alignment(
            np.full_like(alignment, fill_value=4)
        )
