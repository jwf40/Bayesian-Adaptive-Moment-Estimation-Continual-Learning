import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=(1,1))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=(1,1))
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=(1,1))
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=(1,1))
        self.fc1 = nn.Linear(800, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.dropout(self.pool(x), p=0.25)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.dropout(self.pool(x), p=0.25)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.dropout(F.relu(self.fc1(x)), p=0.5)
        x = self.fc2(x)
        return x