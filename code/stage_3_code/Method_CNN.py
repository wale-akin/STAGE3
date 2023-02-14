from code.base_class.method import method
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np


class Method_CNN(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 500
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3