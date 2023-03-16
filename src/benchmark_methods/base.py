import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

class BaseCLMethod:
    def __init__(self, model, train_loader, test_loader, **kwargs):
        self.model = model
        self.train_loader = train_loader
        self.test_loader =test_loader
        self.epochs = kwargs['epochs']
        self.device = kwargs['device']
        self.loss_per_iter = []
        self.test_acc_per_iter = []

    def run(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def save(self):
        pass