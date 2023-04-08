import numpy as np
import random
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, ConcatDataset, Subset
import torchvision
from torchvision.datasets import MNIST
from torchvision.transforms import Compose
import torchvision.transforms as transforms

from tqdm import tqdm
from data import PermutedMNIST
from models.basic_mlp import BasicMLP
from optimizers_lib import fastbgd, bgd
from .base import BaseCLMethod

class Badam(BaseCLMethod):
    def __init__(self, model, train_loader, test_loader, **kwargs):
        super().__init__(model, train_loader, test_loader, \
                         file_name = f"BAdam_ds_{kwargs['exp']}_graduated_{kwargs['graduated']}",**kwargs)
        self.mean_eta=0.4
        self.optim = fastbgd(model, mean_eta=self.mean_eta, std_init=0.06)#bgd(model, mean_eta=mean_eta, std_init=0.06)
        
    def train(self, loader):
        for epoch in tqdm(range(self.epochs)):
            rloss = 0
            for idx,batch in enumerate(tqdm(loader)):
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                mcloss = 0
                for mc_iter in range(self.optim.mc_iters):
                    self.optim.randomize_weights()
                    output = self.model(x)
                    loss = self.criterion(output, y)
                    mcloss += loss
                    #print(loss)
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.aggregate_grads(len(y))
                mcloss/=self.optim.mc_iters
                self.optim.step()
                if not self.use_labels and idx %100==0:
                    self.test()