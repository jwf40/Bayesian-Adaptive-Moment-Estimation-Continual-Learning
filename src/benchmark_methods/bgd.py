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
from optimizers_lib import bgd
from .base import BaseCLMethod

class BGD(BaseCLMethod):
    def __init__(self, model, train_loader, test_loader, **kwargs):
        super().__init__(model, train_loader, test_loader, \
                         file_name = f"BGD_ds_{kwargs['exp']}_graduated_{kwargs['graduated']}",**kwargs)
        self.mean_eta=kwargs['bgd_mean_eta']#kwargs['bgd_mean_eta']
        self.optim = bgd(model, mean_eta=self.mean_eta, std_init=kwargs['bgd_std'])#bgd_std
        
    def train(self, loader):
        xbuffer = []
        ybuffer = []
        for epoch in tqdm(range(self.epochs)):
            rloss = 0
            for idx,batch in enumerate(tqdm(loader)):
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                xbuffer.append(x)
                ybuffer.append(y)
                if len(ybuffer) >= self.buffer_len: 
                    _x = torch.stack(xbuffer).squeeze(1) if self.buffer_len > 1 else x
                    _y = torch.stack(ybuffer).squeeze(1) if self.buffer_len > 1 else y 
                    mcloss = 0
                    for mc_iter in range(self.optim.mc_iters):
                        self.optim.randomize_weights()
                        output = self.model(_x)
                        loss = self.criterion(output, _y)
                        mcloss += loss
                        #print(loss)
                        self.optim.zero_grad()
                        loss.backward()
                        self.optim.aggregate_grads(len(y))
                    mcloss/=self.optim.mc_iters
                    self.optim.step()
                    xbuffer = []
                    ybuffer = []
                # if b_idx%20==0:
                #     running_accs.append(test(model,test_loaders,device))  
                #     print(running_accs[-1])
                if not self.use_labels and idx %self.test_every==0:
                    self.test()


