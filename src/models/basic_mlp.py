import torch
import torch.nn as nn

class BasicMLP(nn.Module):
    def __init__(self, hidden=200,n_classes=10,n_samples=50):
        #hidden layers used to be 200    
        super(BasicMLP,self).__init__()
        self.model = nn.Sequential(nn.Linear(in_features=28*28, out_features=hidden),\
                                    nn.ReLU(),
                                    nn.Linear(in_features=hidden, out_features=hidden),
                                    nn.ReLU(),
                                    nn.Linear(in_features=hidden, out_features=n_classes))

    def forward(self,x):
        x = self.model(x)
        return x

