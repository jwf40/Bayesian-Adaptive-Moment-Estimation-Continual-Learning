import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

class BaseCLMethod:
    def __init__(self, model, train_loader, test_loader, **kwargs):
        self.model = model.to(kwargs['device'])
        self.train_loader = train_loader
        self.test_loader =test_loader
        self.epochs = kwargs['epochs']
        self.device = kwargs['device']       
        self.use_labels = kwargs['labels']
        self.optim = optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9,0.999))
        self.criterion = nn.CrossEntropyLoss()
        self.loss_per_iter = []
        self.test_acc_per_iter = []
        self.task_counter = 0

    def run(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def zerolike_params_dict(self):
        return dict([(n, torch.zeros_like(p))
                for n, p in self.model.named_parameters()])
    
    def copy_params_dict(self):
        return dict([(n, p.data.clone())
                for n, p in self.model.named_parameters()])