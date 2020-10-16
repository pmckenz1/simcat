#!/usr/bin/env python

"""
Code for batch training a machine learning model.
"""

import h5py
import numpy as np
import os
import toytree
import pandas as pd

from keras.utils import to_categorical
from keras.models import load_model
from keras.utils import Sequence


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
                 ):
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

        self.counts_filepath = os.path.join(directory, input_name+'.counts.h5')
        self.labs_filepath = os.path.join(directory, input_name+'.labels.h5')

        if not self.exists:
            self.write_ref_files()
        else:
            self.load()


    def write_ref_files(self):

        # get total simulations to include

        countsfile = h5py.File(self.counts_filepath,'r')
        labsfile = h5py.File(self.labs_filepath,'r')

        sister_idxs = get_sister_idxs(toytree.tree(countsfile.attrs['tree']))

        num_full_dat = countsfile['counts'].shape[0]

        print(str(num_full_dat) + " total simulations.")

        all_viable_idxs = np.array(range(num_full_dat))

        is_sister_bool = np.array([list(scen) in sister_idxs for scen in np.sort(labsfile['admixture'][:,:2].astype(int))])
        exclude_mag_bool = labsfile['admixture'][:,3] < self.exclude_magnitude

        keeper_idxs_mask = ~(is_sister_bool + exclude_mag_bool)

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
        an_file= h5py.File(self.analysis_filepath,'w')

        an_file.create_dataset('viable',shape=all_viable_idxs.shape,data=all_viable_idxs)
        an_file.create_dataset('training',shape=training_idxs.shape,data=training_idxs)
        an_file.create_dataset('testing',shape=testing_idxs.shape, data=testing_idxs)

        # make one-hot dictionary
        str_categories = []
        for i in all_viable_idxs:
            str_categories.append(','.join(labsfile['admixture'][i][:2].astype(int).astype(str)))

        unique_labs = np.unique(str_categories)
        unique_labs_ints = np.array(range(len(unique_labs))).astype(int)

        self.num_classes = len(unique_labs)
        an_file.attrs['num_classes'] = self.num_classes

        self.input_shape = np.prod(countsfile['counts'].shape[1:])
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

        an_file.close()
        countsfile.close()
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


    def write_onehot_file(self):
        '''
        To write a file that contains the one-hot-encoded labels. Optional
        because this might take lots of space.
        '''
        pass

    def init_model(self,model):
        self.model = model
        self.model_path = os.path.join(self.directory,self.output_name+".model.h5")
        model.save(self.model_path)
        print("New neural network saved to " + self.model_path)

    def load_model(self):
        self.model_path = os.path.join(self.directory,self.output_name+".model.h5")
        self.model = load_model(self.model_path)

    def train(self,
              batch_size,
              num_epochs,
              workers=4):
        countsfile = h5py.File(self.counts_filepath, 'r')
        an_file = h5py.File(self.analysis_filepath, 'r')

        n_classes = an_file.attrs['num_classes']

        labels = dict(zip(an_file['labels'][:, 0], an_file['labels'][:, 1]))
        #training_batch_generator = My_Custom_Generator(np.array(an_file['training']),
        #                                               batch_size,
        #                                               an_file,
        #                                               countsfile)

        training_batch_generator = DataGenerator(np.array(an_file['training']),
                                                 labels,
                                                 countsfile,
                                                 n_classes,
                                                 batch_size
                                                 )

        validation_batch_generator = DataGenerator(np.array(an_file['testing']),
                                                   labels,
                                                   countsfile,
                                                   n_classes,
                                                   batch_size
                                                   )

        #validation_batch_generator = My_Custom_Generator(np.array(an_file['testing']),
        #                                                 batch_size,
        #                                                 an_file,
        #                                                 countsfile)

        # Train model on dataset
        self.model.fit_generator(generator=training_batch_generator,
                                 validation_data=validation_batch_generator,
                                 use_multiprocessing=True,
                                 workers=6,
                                 verbose=1,
                                 epochs=num_epochs)

        #self.model.fit_generator(generator=training_batch_generator,
        #                         steps_per_epoch=int(an_file['training'].shape[0] // batch_size),
        #                         epochs=num_epochs,
        #                         verbose=1,
        #                         validation_data=validation_batch_generator,
        #                         validation_steps=int(an_file['testing'].shape[0] // batch_size))

        self.model.save(self.model_path)

        countsfile.close()
        an_file.close()


class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self,
                 list_IDs,
                 labels,
                 data_file,
                 n_classes,
                 batch_size=32,
                 shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.labels = labels
        self.data_file = data_file
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        y = np.empty((self.batch_size), dtype=int)
        
        X = np.array([self.data_file['counts'][_] for _ in list_IDs_temp])
        X = X.reshape(X.shape[0], -1)
        X = X / X.max()
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

class My_Custom_Generator(Sequence):

    def __init__(self, database_idxs, batch_size, analysis_file, counts_file) :
        self.database_idxs = database_idxs
        self.batch_size = batch_size
        self.analysis_file = analysis_file
        self.counts_file = counts_file

    def __len__(self):
        return (np.ceil(len(self.database_idxs) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_idxs = self.database_idxs[idx * self.batch_size : (idx+1) * self.batch_size]

        filler = np.zeros((len(batch_idxs)),dtype=np.int)
        counter = 0
        for i_ in batch_idxs:
            filler[counter] = np.argmax(self.analysis_file['labels'][:,0] == i_)
            counter = counter + 1

        batch_y_ints = np.array([self.analysis_file['labels'][_,1] for _ in filler]) 
        batch_y = to_categorical(batch_y_ints,
                                 num_classes=self.analysis_file.attrs['num_classes'])

        batch_x = np.array([self.counts_file['counts'][_] for _ in batch_idxs])
        batch_x = batch_x.reshape(batch_x.shape[0], -1)
        batch_x = batch_x / batch_x.max()

        return batch_x, batch_y

def get_sister_idxs(tre):
    sisters = []
    for node in tre.treenode.traverse():
        if len(node.children) == 2:
            sisters.append(list(np.sort([i.idx for i in node.children])))
    return(sisters)
