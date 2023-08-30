import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from torchvision.models import resnet18
from avalanche.benchmarks.classic import PermutedMNIST, SplitMNIST, SplitFMNIST,SplitCIFAR10,SplitCIFAR100
from avalanche.models import SimpleMLP, MTSimpleMLP, SimpleCNN, MTSimpleCNN
from avalanche.training import Naive, MAS
from models.vcl_model import VCLModel

def cartpole()->dict:
    kwargs = {
'epochs': 20,
        'batch_size': 256,
        'coreset': 40,
        'model': SimpleMLP(num_classes=10, hidden_layers=4, hidden_size=200),
        'dataset': SplitFMNIST(5),
        'vcl_model': VCLModel(num_classes=10, hidden_layers=4, hidden_size=200),
    }
    return kwargs