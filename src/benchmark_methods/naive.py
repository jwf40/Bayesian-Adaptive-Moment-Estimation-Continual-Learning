from collections import defaultdict
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from .base import BaseCLMethod


class Naive(BaseCLMethod):
    def __init__(self, model, train_loader, test_loader, **kwargs):
        super().__init__(model, train_loader, test_loader, \
                         file_name = f"NAIVE_ds_{kwargs['exp']}_graduated_{kwargs['graduated']}", **kwargs)

    def train(self, loader):        
        for ep in tqdm(range(self.epochs)):
            for idx, data in enumerate(tqdm(loader)):
                self.optim.zero_grad()
                x, y = data[0].to(self.device), data[1].to(self.device)
                out = self.model(x)
                loss = self.criterion(out, y)
                loss.backward()
                self.optim.step()

                if not self.use_labels  and idx %self.test_every==0:
                    self.test()