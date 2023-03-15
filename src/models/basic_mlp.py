import torch
import torch.nn as nn

class BasicMLP(nn.Module):
    def __init__(self, n_classes=10,n_samples=50):    
        super(BasicMLP,self).__init__()
        self.model = nn.Sequential(nn.Linear(in_features=28*28, out_features=200),\
                                    nn.LeakyReLU(),
                                    nn.Linear(in_features=200, out_features=200),
                                    nn.LeakyReLU(),
                                    nn.Linear(in_features=200, out_features=n_classes))

    def forward(self,x):
        x = self.model(x)
        return x
