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
from optimizers_lib import fastbgd, bgd, badam
from .base import BaseCLMethod

class BufferBadam(BaseCLMethod):
    def __init__(self, model, train_loader, test_loader, **kwargs):
        super().__init__(model, train_loader, test_loader, \
                         file_name = f"BufferBAdam_ds_{kwargs['exp']}_graduated_{kwargs['graduated']}_eta_{kwargs['new_eta']}_std_{kwargs['new_std']}",**kwargs)
        #raise AssertionError
        self.mean_eta=kwargs['new_eta']#0.3
        self.optim = badam(model, mean_eta=self.mean_eta, std_init=kwargs['new_std'])#badam bgd(model, mean_eta=mean_eta, std_init=0.06)0.06
        
    def train(self, loader):
        buffer_len = 128
        xbuffer = []
        ybuffer = []
        for epoch in tqdm(range(self.epochs)):
            rloss = 0
            for idx,batch in enumerate(tqdm(loader)):
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                xbuffer.append(x)
                ybuffer.append(y)
                if len(ybuffer) >= buffer_len: 
                    _x = torch.stack(xbuffer).squeeze(1)
                    _y = torch.stack(ybuffer).squeeze(1)   
                    mcloss = 0                
                    for mc_iter in range(self.optim.mc_iters):
                        self.optim.randomize_weights()
                        output = self.model(_x)
                        loss = self.criterion(output, _y)
                        mcloss += loss
                        #print(loss)
                        self.optim.zero_grad()
                        loss.backward()
                        self.optim.aggregate_grads(len(_y))
                    mcloss/=self.optim.mc_iters
                    self.optim.step()
                    #empty buffer
                    xbuffer = []
                    ybuffer = []
                if not self.use_labels and idx %5000==0:
                    self.test()
                