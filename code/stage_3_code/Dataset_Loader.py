'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
from code.base_class.setting import setting
import numpy as np
import csv
import pickle
import matplotlib.pyplot as plt


class Dataset_Loader(dataset, setting):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)


    def help_load(self, source_folder, source_file):
        if 1:
            list = []
            label = []
            f = open(source_folder + source_file, 'rb')  # or change MNIST to other dataset names
            data = pickle.load(f)

            f.close()
            print('training set size:', len(data['train']), 'testing set size:', len(data['test']))
            for pair in data['train']:
                training_image = np.asarray(pair['image'])
                training_label = pair['label']
                list.append(training_image)
                label.append(training_label)
            training_list_matrices = np.array(list)  # change feature list into matrix
            training_label_matrix = np.array(label)

            list.clear() # clear for testing data
            label.clear() # clear for testing data
            for pair in data['test']:
                test_image = np.asarray(pair['image'])
                test_label = pair['label']
                list.append(test_image)
                label.append(test_label)
            test_list_matrices = np.array(list)  # change feature list into matrix
            test_label_matrix = np.array(label)
            print("finish loading! \n",)
            return training_list_matrices, training_label_matrix, test_list_matrices, test_label_matrix

    def load(self):
        print('loading data...')
        data = self.help_load(self.dataset_source_folder_path, self.dataset_source_file_name)
        return data


