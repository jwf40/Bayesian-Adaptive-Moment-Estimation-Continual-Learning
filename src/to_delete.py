import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from models.custom_bnn import MLP
from test_optim import BAdam

test_nn = nn.Sequential(nn.Linear(28*28,100),nn.ReLU(),nn.Linear(100,10))

model = nn.Sequential(nn.Linear(28*28,100),nn.ReLU(),nn.Linear(100,10))#MLP(50,3)
opt = BAdam(model.named_parameters(), model.named_buffers(), model.parameters())
opt2 = optim.SGD(model.parameters(),lr=1.01)
criterion = torch.nn.CrossEntropyLoss()


input = torch.randn(28*28)
targets = torch.zeros(10)
targets[2] = 1.0
opt.zero_grad()
output = model(input)
loss = criterion(output, targets)
loss.backward()
opt.set_batch_size(1)
opt.step()
