'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
from code.base_class.setting import setting
import numpy as np
import pickle
import csv


class Dataset_Loader(dataset, setting):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def help_load(self, source_folder, source_file):
        if 1:
            img_list = []
            label_list = []
            f = open(source_folder + source_file, 'rb')  # or change MNIST to other dataset names
            data = pickle.load(f)

            f.close()
            print('training set size:', len(data['train']), 'testing set size:', len(data['test']))
            for pair in data['train']:
                training_image = np.asarray(pair['image'])
                training_label = pair['label']
                # comment out each if you want to see the outputs for yourself
                #                 print(training_label)
                #---------------
                #                 plt.imshow(training_image, cmap="Greys")
                #                 plt.show()
                img_list.append(training_image)
                label_list.append(training_label)
            training_list_matrices = np.array(img_list)  # change feature list into matrix
            training_label_matrix = np.array(label_list)
            img_list = []
            label_list = []
            for pair in data['test']:
                test_image = np.asarray(pair['image'])
                test_label = pair['label']
                #                 print(test_label)
                #----------------
                #                 plt.imshow(test_image, cmap="Greys")
                #                 plt.show()
                img_list.append(test_image)
                label_list.append(test_label)
            test_list_matrices = np.array(img_list)  # change feature list into matrix
            test_label_matrix = np.array(label_list)

            return training_list_matrices, training_label_matrix, test_list_matrices, test_label_matrix

    def load(self):
        print('loading data...')
        training_list_matrices, training_label_matrix, test_list_matrices, test_label_matrix = self.help_load(self.dataset_source_folder_path, self.dataset_source_file_name)
        training_data = [training_list_matrices, training_label_matrix]
        testing_data = [test_list_matrices, test_label_matrix]

        return training_data, testing_data


# data_obj = Dataset_Loader('MNIST','')
# data_obj.dataset_source_folder_path = '/kaggle/input/mnist/'
# data_obj.dataset_source_file_name = 'MNIST'
# null = data_obj.load()
#
# print(null[0])