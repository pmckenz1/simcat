#!/usr/bin/env python

"""
Code for batch training a machine learning model.
"""

import h5py
import numpy as np
import os
import toytree
import pandas as pd
import sqlite3
import io
import itertools
import tempfile
from numba import njit

#from simcat.utils import get_snps_count_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, concatenate
from tensorflow.keras import Input, Model
import tensorflow as tf

from .utils import SimcatError
from .artifacts import (
    FEATURE_NORMALIZATION,
    MODEL_SCHEMA_VERSION,
    dependency_versions,
    read_database_metadata,
    read_model_metadata,
    write_hdf5_metadata,
    write_model_metadata,
)
from .config import TrainingConfig

class BatchTrain:
    def __init__(self,
                 input_name,
                 output_name,
                 directory,
                 exists=False,
                 prop_training=0.9,
                 exclude_sisters=True,
                 exclude_magnitude=0.1,
                 to_zero_magnitude=0,
                 directionality=True,
                 exclude_mask=None,
                 seed=None,
                 ):
        '''
        exclude_mask: np.array (bool).
            specifies certain rows to exclude -- maybe because they aren't done.
        '''
        self.input_name = input_name
        self.output_name = output_name
        self.directory = directory
        self.exists = exists

        self.prop_training = prop_training
        self.exclude_sisters = exclude_sisters
        self.exclude_magnitude = exclude_magnitude
        self.to_zero_magnitude = to_zero_magnitude
        self.directionality = directionality
        self.exclude_mask = exclude_mask
        self.seed = seed

        if not 0 < prop_training < 1:
            raise ValueError("prop_training must be strictly between 0 and 1")
        if exclude_magnitude < 0:
            raise ValueError("exclude_magnitude must be non-negative")
        if to_zero_magnitude:
            raise NotImplementedError(
                "to_zero_magnitude is not implemented in simcat 0.0.7; "
                "use zero to preserve published category semantics"
            )
        if not directionality:
            raise NotImplementedError(
                "directionality=False is not implemented in simcat 0.0.7"
            )

        self.model = None
        self.newick = None
        self.admixture_row = None

        self.counts_filepath = os.path.join(directory, input_name+'.counts.h5')
        self.labs_filepath = os.path.join(directory, input_name+'.labels.h5')

        if not os.path.exists(self.labs_filepath):
            raise FileNotFoundError(
                f"Labels database not found: {self.labs_filepath}"
            )
        self.database_metadata = read_database_metadata(self.labs_filepath)
        sql_path = os.path.join(directory, input_name + ".counts.db")
        if os.path.exists(sql_path) and self._counts_need_sync(sql_path):
            print("Synchronizing simulated counts from SQLite to HDF5...")
            self.write_sql_counts_to_h5()
        elif not os.path.exists(self.counts_filepath):
            raise FileNotFoundError(
                "Counts data were not found. Expected either "
                f"{self.counts_filepath} or {sql_path}."
            )

        self.analysis_filepath = os.path.join(self.directory,self.output_name+'.analysis.h5')
        if not os.path.exists(self.analysis_filepath):
            self.write_ref_files()
        else:
            self._validate_analysis_completion_state()
            self.load()

    @classmethod
    def from_config(cls, config):
        """Create a trainer from a validated :class:`TrainingConfig`."""
        if not isinstance(config, TrainingConfig):
            raise TypeError("config must be a simcat.config.TrainingConfig")
        instance = cls(
            input_name=config.input_name,
            output_name=config.output_name,
            directory=config.directory,
            prop_training=config.prop_training,
            exclude_sisters=config.exclude_sisters,
            exclude_magnitude=config.exclude_magnitude,
            seed=config.seed,
        )
        instance.training_config = config
        return instance

    def _validate_analysis_completion_state(self):
        """Reject a split file made before newly completed simulations."""
        with h5py.File(self.analysis_filepath, "r") as analysis:
            recorded = analysis.attrs.get("completed_simulations")
        if recorded is None:
            # Legacy analysis artifacts did not record this value. Preserve
            # their load behavior; Phase 2 will provide explicit migration.
            return
        with h5py.File(self.labs_filepath, "r") as labels:
            current = int(
                np.count_nonzero(np.asarray(labels["finished_sims"]) == 1)
            )
        if int(recorded) != current:
            raise SimcatError(
                f"Analysis metadata records {int(recorded)} completed "
                f"simulations, but the labels database now has {current}. "
                "Choose a new output_name or remove the stale analysis/model "
                "artifacts before creating a new training split."
            )

    def _counts_need_sync(self, sql_path):
        """Return whether the training HDF5 is absent or older than SQLite."""
        if not os.path.exists(self.counts_filepath):
            return True
        try:
            with h5py.File(self.counts_filepath, "r") as countsfile:
                synchronized = bool(
                    countsfile.attrs.get("sqlite_synchronized", False)
                )
                recorded_mtime = int(
                    countsfile.attrs.get("sqlite_mtime_ns", -1)
                )
        except OSError:
            return True
        return (
            not synchronized
            or recorded_mtime != os.stat(sql_path).st_mtime_ns
        )


    def write_sql_counts_to_h5(self):
        """Atomically synchronize SQLite simulation arrays to training HDF5."""
        sql_path = os.path.join(self.directory, self.input_name+'.counts.db')
        with h5py.File(self.labs_filepath, "r") as labsfile:
            num_full_dat = labsfile["finished_sims"].shape[0]
            finished_states = np.asarray(labsfile["finished_sims"])
            countshape = (
                int(labsfile.attrs["ntips"]),
                int(labsfile.attrs["nsnps"]),
            )
            metadata = dict(labsfile.attrs.items())

        fd, temp_path = tempfile.mkstemp(
            prefix=f".{self.input_name}.counts-",
            suffix=".h5.tmp",
            dir=self.directory,
        )
        os.close(fd)
        try:
            con = sqlite3.connect(
                sql_path, detect_types=sqlite3.PARSE_DECLTYPES
            )
            try:
                cur = con.cursor()
                with h5py.File(temp_path, mode="w") as out:
                    counts = out.create_dataset(
                        name="counts",
                        shape=(num_full_dat, *countshape),
                        dtype=np.int64,
                        compression="gzip",
                    )
                    for key, value in metadata.items():
                        out.attrs[key] = value
                    for simulation_number in range(num_full_dat):
                        cur.execute(
                            "select arr from counts where id=?",
                            (simulation_number,),
                        )
                        data = cur.fetchone()
                        if data is None:
                            raise SimcatError(
                                "SQLite counts table is missing row "
                                f"{simulation_number}."
                            )
                        if data[0] is None:
                            if finished_states[simulation_number] == 1:
                                raise SimcatError(
                                    "Simulation row "
                                    f"{simulation_number} is marked complete "
                                    "but has no SQLite counts array."
                                )
                            # HDF5 datasets are zero-filled by default; pending
                            # and reserved rows are excluded from training.
                            continue
                        array = np.asarray(data[0])
                        if array.shape != countshape:
                            raise SimcatError(
                                f"Counts row {simulation_number} has shape "
                                f"{array.shape}; expected {countshape}."
                            )
                        counts[simulation_number] = array
                    out.attrs["sqlite_synchronized"] = True
                    out.attrs["sqlite_mtime_ns"] = os.stat(sql_path).st_mtime_ns
            finally:
                con.close()
            os.replace(temp_path, self.counts_filepath)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


    def write_ref_files(self):

        # get total simulations to include

        #countsfile = h5py.File(self.counts_filepath,'r')
        labsfile = h5py.File(self.labs_filepath,'r')

        # the last admixture event is the one we're interested in
        self.admixture_row = labsfile['admixture'].shape[1] - 1

        sister_idxs = get_sister_idxs(toytree.tree(labsfile.attrs['tree']))
        self.newick = labsfile.attrs['tree']
        self.nquarts = labsfile.attrs['nquarts']

        num_full_dat = labsfile['finished_sims'].shape[0]

        print(str(num_full_dat) + " total simulations.")

        all_viable_idxs = np.array(range(num_full_dat))

        # which ones are unfinished?
        finished_states = np.asarray(labsfile['finished_sims'])
        is_unfinished_bool = finished_states != 1
        completed_simulations = int(
            np.count_nonzero(finished_states == 1)
        )

        # if exlcuding sisters, which are sisters?
        if self.exclude_sisters:
            is_sister_bool = np.array([list(scen) in sister_idxs for scen in np.sort(labsfile['admixture'][:, self.admixture_row, :2].astype(int))])
        else:  # otherwise call none of them sisters
            is_sister_bool = np.zeros((num_full_dat),dtype=bool)

        # if excluding under a magnitude, which are under that magnitude?
        exclude_mag_bool = labsfile['admixture'][:, self.admixture_row, 3] < self.exclude_magnitude

        keeper_idxs_mask = ~(
            is_unfinished_bool | is_sister_bool | exclude_mag_bool
        )
        if self.exclude_mask is not None:
            exclude_mask = np.asarray(self.exclude_mask, dtype=bool)
            if exclude_mask.shape != (num_full_dat,):
                raise ValueError(
                    "exclude_mask must have one boolean value per simulation"
                )
            keeper_idxs_mask &= ~exclude_mask

        all_viable_idxs = all_viable_idxs[keeper_idxs_mask]

        num_viable = len(all_viable_idxs)
        if num_viable < 2:
            labsfile.close()
            raise SimcatError(
                "At least two completed simulations compatible with the "
                "filters are required for training and validation."
            )
        num_training = min(
            max(int(num_viable * self.prop_training), 1),
            num_viable - 1,
        )
        self.num_training = num_training
        self.num_testing = num_viable - num_training

        print(str(num_viable) + " total simulations compatible with parameters.")
        print("Data split into " + str(self.num_training) + " training and " + str(self.num_testing) + " testing simulations.")

        random = np.random.RandomState(self.seed)
        training_idxs = np.sort(
            random.choice(all_viable_idxs, num_training, replace=False)
        )
        testing_idxs = np.sort(np.array(list(set(all_viable_idxs).difference(set(training_idxs)))))

        self.analysis_filepath = os.path.join(self.directory,self.output_name+'.analysis.h5')
        an_file = h5py.File(self.analysis_filepath, 'w')

        an_file.create_dataset('viable', shape=all_viable_idxs.shape,data=all_viable_idxs)
        an_file.create_dataset('training', shape=training_idxs.shape,data=training_idxs)
        an_file.create_dataset('testing', shape=testing_idxs.shape, data=testing_idxs)

        # make one-hot dictionary
        str_categories = []
        for i in all_viable_idxs:
            str_categories.append(','.join(labsfile['admixture'][i][self.admixture_row, :2].astype(int).astype(str)))

        unique_labs = np.unique(str_categories)
        unique_labs_ints = np.array(range(len(unique_labs))).astype(int)

        self.num_classes = len(unique_labs)
        an_file.attrs['num_classes'] = self.num_classes

        self.input_shape = (self.nquarts, 16 * 16)
        an_file.attrs['input_shape'] = self.input_shape

        self.onehot_dict_path = os.path.join(self.directory,self.output_name+'.onehot_dict.csv')
        pd.DataFrame([unique_labs_ints, unique_labs]).to_csv(self.onehot_dict_path,
                                                             index=False)
        print('')
        print('Onehot dictionary file saved to ' + self.onehot_dict_path)

        inv_onehot_dict = dict(zip(unique_labs,range(len(unique_labs))))

        y_ints = [inv_onehot_dict[i] for i in np.array(str_categories)]

        an_file.create_dataset('labels',shape=(len(y_ints),2), data=np.array([all_viable_idxs,y_ints]).T)

        # add the other attributes from __init__
        an_file.attrs['prop_training'] = self.prop_training
        an_file.attrs['exclude_sisters'] = self.exclude_sisters
        an_file.attrs['exclude_magnitude'] = self.exclude_magnitude
        an_file.attrs['to_zero_magnitude'] = self.to_zero_magnitude
        an_file.attrs['directionality'] = self.directionality
        an_file.attrs['num_training'] = self.num_training
        an_file.attrs['num_testing'] = self.num_testing
        an_file.attrs['newick'] = self.newick
        an_file.attrs['nquarts'] = self.nquarts
        an_file.attrs['seed'] = -1 if self.seed is None else int(self.seed)
        an_file.attrs['completed_simulations'] = completed_simulations
        analysis_metadata = {
            "artifact_type": "simcat-analysis",
            "schema_version": MODEL_SCHEMA_VERSION,
            "feature_schema_version": self.database_metadata.get(
                "feature_schema_version", 0
            ),
            "feature_normalization": FEATURE_NORMALIZATION,
            "tree_newick": self.newick,
            "tip_order": self.database_metadata.get("tip_order", []),
            "quartet_order": self.database_metadata.get("quartet_order", []),
            "edge_category_map": {
                str(index): label for index, label in enumerate(unique_labs)
            },
            "seeds": {"split": self.seed, "model": self.seed},
            "configuration": {
                "prop_training": self.prop_training,
                "exclude_sisters": bool(self.exclude_sisters),
                "exclude_magnitude": self.exclude_magnitude,
                "to_zero_magnitude": self.to_zero_magnitude,
                "directionality": bool(self.directionality),
            },
            "package_versions": dependency_versions(("tensorflow", "pandas")),
        }
        write_hdf5_metadata(an_file, analysis_metadata)

        an_file.close()
        labsfile.close()

        print('')
        print('Analysis reference file saved to ' + self.analysis_filepath)

    def load(self):
        self.analysis_filepath = os.path.join(self.directory,self.output_name+'.analysis.h5')
        self.onehot_dict_path = os.path.join(self.directory,self.output_name+'.onehot_dict.csv')

        # load in attributes
        with h5py.File(self.analysis_filepath, 'r') as an_file:
            self.num_classes = an_file.attrs['num_classes']
            self.input_shape = an_file.attrs['input_shape']
            self.prop_training = an_file.attrs['prop_training']
            self.exclude_sisters = an_file.attrs['exclude_sisters']
            self.exclude_magnitude = an_file.attrs['exclude_magnitude']
            self.to_zero_magnitude = an_file.attrs['to_zero_magnitude']
            self.directionality = an_file.attrs['directionality']
            self.num_training = an_file.attrs['num_training']
            self.num_testing = an_file.attrs['num_testing']
            self.newick = an_file.attrs['newick']
            self.nquarts = an_file.attrs['nquarts']
            stored_seed = int(an_file.attrs.get('seed', -1))
            self.seed = None if stored_seed == -1 else stored_seed


    def init_model(self, dropout=True, extra_layer=False, force=False, save=True):
        if self.num_classes < 2:
            raise SimcatError(
                "At least two edge categories are required to initialize a model."
            )
        if self.seed is not None:
            tf.keras.utils.set_random_seed(self.seed)
        self.model_path = os.path.join(self.directory, self.output_name + ".model.h5")
        if not os.path.exists(self.model_path) or force:
            nnodes_per_quart = 8  # or make this tunable later

            quart_inputs = [
                Input(shape=(16 * 16,), name=f"input_{i + 1}") for i in range(self.nquarts)
            ]
            x = [Dense(nnodes_per_quart, activation="relu")(quart) for quart in quart_inputs]

            if dropout:
                x = [Dropout(0.5)(layer) for layer in x]

            combined = concatenate(x)

            if extra_layer:
                combined = Dense(self.num_classes, activation='relu')(combined)
                if dropout:
                    combined = Dropout(0.5)(combined)

            outputs = Dense(self.num_classes, activation='softmax')(combined)
            self.model = Model(inputs=quart_inputs, outputs=outputs)

            self.model.compile(loss='categorical_crossentropy',
                               optimizer='adam',
                               metrics=['accuracy'])

            if save:
                self.model.save(self.model_path)
                self._write_model_metadata(dropout, extra_layer)
                print("New neural network saved to:", self.model_path)
        else:
            print("Model already exists. Load it using `load_model()`.")

    def load_model(self):
        self.model_path = os.path.join(self.directory, self.output_name + ".model.h5")
        if os.path.exists(self.model_path):
            # Schema-0 models remain readable; callers can require or migrate a
            # current sidecar through simcat.artifacts before loading.
            self.model_metadata = read_model_metadata(
                self.model_path,
                analysis_path=self.analysis_filepath,
                categories_path=self.onehot_dict_path,
            )
            print("Loading existing neural network:", self.model_path)
            self.model = load_model(self.model_path)

            # Always explicitly recompile after loading
            self.model.compile(loss='categorical_crossentropy',
                               optimizer='adam',
                               metrics=['accuracy'])
        else:
            raise FileNotFoundError("Model file not found. Use `init_model()` to create one.")

    def train(self, batch_size, num_epochs):
        if self.model is None:
            raise SimcatError(
                "No neural network is loaded. Call init_model() for a new "
                "model or load_model() for an existing model before train()."
            )
        if int(batch_size) != batch_size or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if int(num_epochs) != num_epochs or num_epochs <= 0:
            raise ValueError("num_epochs must be a positive integer")
        with h5py.File(self.analysis_filepath, 'r') as an_file:
            training_ids = an_file['training'][:]
            testing_ids = an_file['testing'][:]
            labels_dict = dict(an_file['labels'][:])
            n_classes = an_file.attrs['num_classes']

        tree = toytree.tree(self.newick)
        counts_h5 = h5py.File(self.counts_filepath, 'r')

        def load_from_h5(sample_id):
            sample_id = int(sample_id)
            raw_counts = counts_h5['counts'][sample_id]
            processed = get_snps_count_matrix(tree, raw_counts)
            processed = processed.reshape(self.nquarts, -1).astype('float32')
            processed = _normalize_count_matrices(processed)
            label = to_categorical(labels_dict[sample_id], num_classes=n_classes)
            return tuple([processed[i] for i in range(self.nquarts)] + [label])

        def tf_load(sample_id):
            output_types = tuple([tf.float32] * self.nquarts + [tf.float32])
            result = tf.py_function(load_from_h5, [sample_id], Tout=output_types)
            inputs = {f'input_{i+1}': result[i] for i in range(self.nquarts)}
            label = result[-1]
            for i in range(self.nquarts):
                inputs[f'input_{i+1}'].set_shape([256])
            label.set_shape([self.num_classes])
            return inputs, label

        ds_train = (tf.data.Dataset.from_tensor_slices(training_ids)
                    .map(tf_load, num_parallel_calls=tf.data.AUTOTUNE)
                    .batch(batch_size)
                    .prefetch(tf.data.AUTOTUNE))

        ds_val = (tf.data.Dataset.from_tensor_slices(testing_ids)
                  .map(tf_load, num_parallel_calls=tf.data.AUTOTUNE)
                  .batch(batch_size)
                  .prefetch(tf.data.AUTOTUNE))

        try:
            history = self.model.fit(
                ds_train,
                epochs=int(num_epochs),
                validation_data=ds_val,
                verbose=1,
            )
        finally:
            counts_h5.close()

        # Explicitly save updated model
        self.model.save(self.model_path)
        self._write_model_metadata()
        print("Model trained and saved to:", self.model_path)
        return history

    def _write_model_metadata(self, dropout=None, extra_layer=None):
        """Write the schema sidecar that makes the Keras file interpretable."""
        categories = pd.read_csv(self.onehot_dict_path).iloc[1].tolist()
        configuration = {
            "prop_training": float(self.prop_training),
            "exclude_sisters": bool(self.exclude_sisters),
            "exclude_magnitude": float(self.exclude_magnitude),
            "directionality": bool(self.directionality),
        }
        if dropout is not None:
            configuration["dropout"] = bool(dropout)
        if extra_layer is not None:
            configuration["extra_layer"] = bool(extra_layer)
        metadata = {
            "feature_schema_version": self.database_metadata.get(
                "feature_schema_version", 0
            ),
            "feature_normalization": FEATURE_NORMALIZATION,
            "tree_newick": self.newick,
            "tip_order": self.database_metadata.get("tip_order", []),
            "quartet_order": self.database_metadata.get("quartet_order", []),
            "edge_category_map": {
                str(index): str(category)
                for index, category in enumerate(categories)
            },
            "input_shape": [int(value) for value in self.input_shape],
            "seeds": {"split": self.seed, "model": self.seed},
            "configuration": configuration,
            "package_versions": dependency_versions(("tensorflow", "pandas")),
        }
        write_model_metadata(self.model_path, metadata)
        self.model_metadata = metadata | {
            "artifact_type": "simcat-model",
            "schema_version": MODEL_SCHEMA_VERSION,
        }

    def _format_alignment_for_model(self, alignment):
        '''
        Formatting an alignment of unlinked SNP data for neural net
        Alignment rows MUST match the order in the simcat database (ie from the tree)
        '''
        alignment = np.asarray(alignment)
        tree = toytree.tree(self.newick)
        if alignment.ndim != 2:
            raise ValueError("alignment must be a two-dimensional array")
        if alignment.shape[0] != tree.ntips:
            raise ValueError(
                f"alignment has {alignment.shape[0]} rows; expected "
                f"{tree.ntips} rows in tree-tip order"
            )
        if alignment.shape[1] == 0:
            raise ValueError("alignment must contain at least one SNP")
        if not np.issubdtype(alignment.dtype, np.integer):
            raise ValueError("alignment allele codes must be integers from 0 to 3")
        if np.any((alignment < 0) | (alignment > 3)):
            raise ValueError("alignment allele codes must be integers from 0 to 3")

        # format in quartet matrices
        mat = get_snps_count_matrix(tree, alignment)
        # reshape it to combine the 16x16 part
        mat = mat.reshape(mat.shape[0], 1, -1).astype("float32")
        mat = _normalize_count_matrices(mat)
        # make a dictionary giving each separate quartet matrix an input name
        # and reshaping it the way keras likes (ie with a row dimension)
        counts_dict = {"input_" + str(quart+1): mat[quart] for quart in range(len(mat))}
        return(counts_dict)

    def predict_from_alignment(self, alignment):
        if self.model is None:
            raise SimcatError(
                "No neural network is loaded. Call load_model() or "
                "init_model() before prediction."
            )
        # format the alignment for the model
        count_dict = self._format_alignment_for_model(alignment)

        # load in the onehot dictionary linking model vals to understandable vals
        # the understandable vals are always in numerical indexed order, 0 to max categories
        onehot = pd.read_csv(self.onehot_dict_path)

        # make the prediction
        pred = self.model.predict(count_dict)

        # align with categories in a DataFrame
        pred_df = pd.DataFrame([pred[0]],columns=onehot.loc[1])
        return(pred_df)


def convert_array(text):
    '''
    converts bytes array in sql db to numpy
    '''
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out, allow_pickle=False)


sqlite3.register_converter("array", convert_array)


def _normalize_count_matrices(matrices):
    """Apply the published per-quartet max normalization with zero guards."""
    matrices = np.asarray(matrices, dtype=np.float32)
    maxima = matrices.max(axis=-1, keepdims=True)
    if np.any(maxima <= 0):
        raise SimcatError(
            "A quartet count matrix is empty and cannot be normalized."
        )
    return matrices / maxima

def get_sister_idxs(tre):
    sisters = []
    for node in tre.treenode.traverse():
        if len(node.children) == 2:
            sisters.append(list(np.sort([i.idx for i in node.children])))
    return(sisters)


def get_snps_count_matrix(tree, seqs):
    """
    Compiles SNP data into a nquartets x 16 x 16 count matrix with the order
    of quartets determined by the shape of the tree.
    """
    # get all quartets for this size tree
    quarts = list(itertools.combinations(range(tree.ntips), 4))

    # shape of the arr (count matrix)
    arr = np.zeros((len(quarts), 16, 16), dtype=np.int64)

    # iterator for quartets, e.g., (0, 1, 2, 3), (0, 1, 2, 4)...
    quartidx = 0
    for currquart in quarts:
        # cols indices match tip labels b/c we named tips node.idx
        quartsnps = seqs[currquart, :]
        # save as stacked matrices
        arr[quartidx] = count_matrix_int(quartsnps)
        # save flattened to counts
        quartidx += 1
    return arr


@njit
def count_matrix_int(quartsnps):
    """
    return a 16x16 matrix of site counts from snparr
    """
    arr = np.zeros((16, 16), dtype=np.int64)
    for idx in range(quartsnps.shape[1]):
        i = quartsnps[:, idx]
        arr[(4 * i[0]) + i[1], (4 * i[2]) + i[3]] += 1
    return arr
