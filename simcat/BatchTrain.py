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

from simcat.utils import get_snps_count_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Dense, Dropout, concatenate
from tensorflow.keras import Input, Model

import tensorflow as tf
import sqlite3
import numpy as np
from tensorflow.keras.utils import to_categorical

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

        self.model = None
        self.newick = None
        self.admixture_row = None

        self.counts_filepath = os.path.join(directory, input_name+'.counts.h5')
        self.labs_filepath = os.path.join(directory, input_name+'.labels.h5')

        if not os.path.exists(self.counts_filepath):
            if os.path.exists(os.path.join(directory, input_name+'.counts.db')):
                print("hdf5 counts file does not yet exist. Converting SQL database to hdf5...")
                self.write_sql_counts_to_h5()

        self.analysis_filepath = os.path.join(self.directory,self.output_name+'.analysis.h5')
        if not os.path.exists(self.analysis_filepath):
            self.write_ref_files()
        else:
            self.load()

    def write_sql_counts_to_h5(self):
        sql_path = os.path.join(self.directory, self.input_name+'.counts.db')
        labsfile = h5py.File(self.labs_filepath,'r')
        num_full_dat = labsfile['finished_sims'].shape[0]
        labsfile.close()

        # get the alignment shape
        con = sqlite3.connect(sql_path, detect_types=sqlite3.PARSE_DECLTYPES)
        cur = con.cursor()

        cur.execute("select arr from counts where id={}".format(0))
        data = cur.fetchone()
        countshape = data[0].shape

        con.close()

        o5 = h5py.File(self.counts_filepath, mode='w')
        o5.create_dataset(name="counts",
                          shape=(num_full_dat,
                                 countshape[0],
                                 countshape[1]),
                          dtype=np.int64)

        con = sqlite3.connect(sql_path, detect_types=sqlite3.PARSE_DECLTYPES)
        cur = con.cursor()

        for simulation_number in range(num_full_dat):
            cur.execute("select arr from counts where id={}".format(simulation_number))
            data = cur.fetchone()
            o5['counts'][simulation_number] = data[0]
            #converted = convert_array(data[0])
            #o5['counts'][simulation_number] = converted

        con.close()
        o5.close()


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
        is_unfinished_bool = ~np.array(labsfile['finished_sims']).astype(bool)

        # if exlcuding sisters, which are sisters?
        if self.exclude_sisters:
            is_sister_bool = np.array([list(scen) in sister_idxs for scen in np.sort(labsfile['admixture'][:, self.admixture_row, :2].astype(int))])
        else:  # otherwise call none of them sisters
            is_sister_bool = np.zeros((num_full_dat),dtype=bool)

        # if excluding under a magnitude, which are under that magnitude?
        exclude_mag_bool = labsfile['admixture'][:, self.admixture_row, 3] < self.exclude_magnitude

        keeper_idxs_mask = ~(is_unfinished_bool + is_sister_bool + exclude_mag_bool)

        all_viable_idxs = all_viable_idxs[keeper_idxs_mask]

        num_viable = len(all_viable_idxs)
        num_training = int(num_viable*self.prop_training)
        self.num_training = num_training
        self.num_testing = num_viable - num_training

        print(str(num_viable) + " total simulations compatible with parameters.")
        print("Data split into " + str(self.num_training) + " training and " + str(self.num_testing) + " testing simulations.")

        training_idxs = np.sort(np.random.choice(all_viable_idxs,num_training,replace=False))
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

        an_file.close()
        labsfile.close()

        print('')
        print('Analysis reference file saved to ' + self.analysis_filepath)

    def load(self):
        self.analysis_filepath = os.path.join(self.directory,self.output_name+'.analysis.h5')
        self.onehot_dict_path = os.path.join(self.directory,self.output_name+'.onehot_dict.csv')

        # load in attributes
        an_file = h5py.File(self.analysis_filepath,'r')
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

#    def init_model(self,
#        nnodes_per_quart=8,
#        ):
#        # define the model -- this could putentially be tuned by user
#        quart_inputs = [Input(shape=(16*16,)) for quartidx in range(self.nquarts)]
#        x = [Dense(nnodes_per_quart, activation="relu")(quart) for quart in quart_inputs]
#        combined = concatenate(x)
#        z = Dense(self.num_classes, activation='softmax')(combined)
#
#        model = Model(inputs=quart_inputs,outputs=z)
#
#        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#        self.model = model
#        self.model_path = os.path.join(self.directory,self.output_name+".model.h5")
#        model.save(self.model_path)
#        print("New neural network saved to " + self.model_path)

    def init_model(self,
        dropout=True,
        extra_layer=False,
        force=False
        ):
        self.model_path = os.path.join(self.directory,self.output_name+".model.h5")
        if not os.path.exists(self.model_path) or force:
            nnodes_per_quart = 8 # this can be tuned by user in the future?
            # define the model -- this could putentially be tuned by user
            quart_inputs = [Input(shape=(16*16,), name="input_" + str(i+1)) for i in range(self.nquarts)]
            x = [Dense(nnodes_per_quart, activation="relu")(quart) for quart in quart_inputs]
            if dropout:
                # add dropout to each input
                x = [Dropout(0.5)(i) for i in x]
            combined = concatenate(x)
            if extra_layer:
                combined = Dense(self.num_classes,activation='relu')(combined) # as many nodes as outputs here - arbitraryBa
                if dropout:
                    combined = Dropout(0.5)(combined)
            z = Dense(self.num_classes, activation='softmax')(combined)

            model = Model(inputs=quart_inputs,outputs=z)

            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            self.model = model

            model.save(self.model_path)
            print("New neural network saved to: " + self.model_path)
        else:
            print("Model already defined -- use load_model() to import")

    def load_model(self):
        self.model_path = os.path.join(self.directory,self.output_name+".model.h5")
        if not os.path.exists(self.model_path):
            print("No model with this output name yet defined -- initialize a model with init_model()")
        else:
            print("Loading existing neural network: " + self.model_path)
            self.model = load_model(self.model_path)


    def get_data(self,
        batch_idxs # list of indices you want to pull data from
        ):

        countsfile = h5py.File(self.counts_filepath, 'r')
        an_file = h5py.File(self.analysis_filepath, 'r')

        n_classes = an_file.attrs['num_classes']

        labels = dict(zip(an_file['labels'][:, 0], an_file['labels'][:, 1]))

        newick = an_file.attrs['newick']

        tree = toytree.tree(newick)

        # Initialization
        y = np.empty((len(batch_idxs)), dtype=int)

        X_ = np.array([countsfile['counts'][_] for _ in batch_idxs])
        X = np.zeros(shape=(X_.shape[0], self.nquarts, 16, 16), dtype=float)
        for row in range(X.shape[0]):
            X[row] = get_snps_count_matrix(tree, X_[row])
            #X[row] = np.array([get_snps_count_matrix(tree, X_[row])])
        #X = X.reshape(X.shape[0], -1)
        #maxes_vector = np.max(X, axis=1) # finds max of each row
        # dividing each row by its max, slicing per: 
        # https://stackoverflow.com/questions/19602187/numpy-divide-each-row-by-a-vector-element
        #X = X / maxes_vector[:, None]

        # Generate data
        for i, ID in enumerate(batch_idxs):
            # Store class
            y[i] = labels[ID]

        countsfile.close()
        an_file.close()

        return X, to_categorical(y, num_classes=n_classes)

    def _format_alignment_for_model(self, alignment):
        '''
        Formatting an alignment of unlinked SNP data for neural net
        Alignment rows MUST match the order in the simcat database (ie from the tree)
        (This order is alphabetical from tip names)
        '''
        # format in quartet matrices
        mat = np.array([get_snps_count_matrix(toytree.tree(self.newick), alignment)])[0]
        # reshape it to combine the 16x16 part
        mat = mat.reshape(mat.shape[0],1,-1)
        mat = mat / np.max(mat,axis=2)[:,np.newaxis]
        # make a dictionary giving each separate quartet matrix an input name
        # and reshaping it the way keras likes (ie with a row dimension)
        counts_dict = {"input_" + str(quart+1): mat[quart] for quart in range(len(mat))}
        return(counts_dict)

    def predict_from_alignment(self, alignment):
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


    # Example new train method for BatchTrain class using tf.data:
    def train(self, batch_size, num_epochs):
        import tensorflow as tf
        print("TensorFlow version:", tf.__version__)
        print("tf.keras version:", tf.keras.__version__)
        print("Executing eagerly:", tf.executing_eagerly())
        if not tf.executing_eagerly():
            tf.compat.v1.enable_eager_execution()

        # Open analysis file to get indices and attributes
        an_file = h5py.File(self.analysis_filepath, 'r')
        training_ids = np.array(an_file['training'])
        testing_ids = np.array(an_file['testing'])
        n_classes = an_file.attrs['num_classes']
        nquarts = an_file.attrs['nquarts']
        newick = an_file.attrs['newick']
        an_file.close()

        # Build the tree and labels dict (as before)
        tree = toytree.tree(newick)
        # Load labels from analysis file
        an_file = h5py.File(self.analysis_filepath, 'r')
        labels = dict(zip(an_file['labels'][:, 0], an_file['labels'][:, 1]))
        an_file.close()

        # Define the SQL path (as in __init__)
        sql_path = os.path.join(self.directory, self.input_name + '.counts.db')

        # Create a tf.data.Dataset for training IDs
        ds_train = tf.data.Dataset.from_tensor_slices(training_ids)
        # Map each sample ID to its processed inputs and label.
        ds_train = ds_train.map(
            lambda sid: map_func(sid, sql_path, tree, labels, nquarts, n_classes),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        ds_train = ds_train.batch(batch_size)
        ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

        # Do the same for validation/test dataset
        ds_val = tf.data.Dataset.from_tensor_slices(testing_ids)
        ds_val = ds_val.map(
            lambda sid: map_func(sid, sql_path, tree, labels, nquarts, n_classes),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        ds_val = ds_val.batch(batch_size)
        ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

        # Train the model for the specified number of epochs.
        for epoch in range(num_epochs):
            print("Epoch {}/{}".format(epoch + 1, num_epochs))
            self.model.fit(
                ds_train,
                epochs=1,
                validation_data=ds_val,
                verbose=1
            )
            self.model.save(self.model_path)


# Define a helper that loads and processes one sample from the database.
# This function uses the get_snps_count_matrix() and mimics the __data_generation logic from previous versions
def load_sample_py(sample_id, sql_path, tree, labels, nquarts, n_classes):
    sample_id = int(sample_id)
    # Open a new connection (each call gets its own connection)
    con = sqlite3.connect(sql_path, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    # Execute query to get the sample from the SQL DB
    result = cur.execute("select arr from counts where id={}".format(sample_id)).fetchone()
    con.close()
    
    # 'arr' should be a numpy array with shape (nquarts, 16, 16)
    arr = result[0]
    # Process the raw counts
    processed = get_snps_count_matrix(tree, arr)  # expected shape: (nquarts, 16, 16)
    # Reshape each quartet to a vector (256 = 16*16)
    processed = processed.reshape(nquarts, -1)  # shape: (nquarts, 256)
    # Normalize each quartet individually
    max_vals = np.max(processed, axis=1, keepdims=True)
    processed = processed / max_vals

    # Get the label for this sample and convert to one-hot
    label_val = labels[sample_id]
    one_hot = to_categorical(label_val, num_classes=n_classes)

    # Instead of returning a dict directly, return a tuple with one tensor per input.
    # Later we convert these into a dict with keys matching the model inputs.
    outputs = tuple([processed[i] for i in range(nquarts)] + [one_hot])
    return outputs

# Wrapper mapping function that uses tf.py_function.
def map_func(sample_id, sql_path, tree, labels, nquarts, n_classes):
    # Specify output types for each element:
    # one tf.float32 tensor per input (nquarts total) and one for the label.
    output_types = tuple([tf.float32] * nquarts + [tf.float32])
    # Call load_sample_py via tf.py_function.
    outputs = tf.py_function(
        func=lambda sid: load_sample_py(sid, sql_path, tree, labels, nquarts, n_classes),
        inp=[sample_id],
        Tout=output_types
    )
    # Set static shapes so downstream layers know what to expect.
    input_tensors = {}
    for i in range(nquarts):
        tensor = outputs[i]
        tensor.set_shape([256])
        input_tensors[f'input_{i+1}'] = tensor
    label_tensor = outputs[-1]
    label_tensor.set_shape([n_classes])
    return input_tensors, label_tensor


def convert_array(text):
    '''
    converts bytes array in sql db to numpy
    '''
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

def get_sister_idxs(tre):
    sisters = []
    for node in tre.treenode.traverse():
        if len(node.children) == 2:
            sisters.append(list(np.sort([i.idx for i in node.children])))
    return(sisters)
