'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD
from code.base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np


class Setting_Train_Test(setting):
    fold = 3

    def load_run_save_evaluate(self):
        # load dataset
        loaded_data = self.dataset.load()
        X_train = loaded_data[0]
        y_train = loaded_data[1]
        X_test = loaded_data[2]
        y_test = loaded_data[3]
        X_train = np.expand_dims(X_train, axis=1)  # match dimensions for conv2d
        X_test = np.expand_dims(X_test, axis=1)  # match dimensions fo conv2d

        # X_train, X_test, y_train, y_test = train_test_split(loaded_data['X'], loaded_data['y'], test_size = 0.33)

        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        return self.evaluate.evaluate(), None
