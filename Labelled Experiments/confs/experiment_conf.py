import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from torchvision.models import resnet18
from avalanche.benchmarks.classic import PermutedMNIST, SplitMNIST, SplitFMNIST,SplitCIFAR10,SplitCIFAR100
from avalanche.models import SimpleMLP, MTSimpleMLP, SimpleCNN, MTSimpleCNN
from avalanche.training import Naive, MAS
from models.vcl_model import VCLModel

def pmnist()->dict:
    kwargs = {
        'epochs': 30,
        'batch_size': 256,
        'coreset': 200,
        'model': SimpleMLP(num_classes=10, hidden_layers=2, hidden_size=100),#MTSimpleMLP(hidden_size=512),#SimpleMLP(num_classes=10, hidden_layers=2, hidden_size=256),SimpleMLP(num_classes=10, hidden_layers=2, hidden_size=256),#
        'vcl_model': VCLModel(num_classes=10, hidden_layers=2, hidden_size=100),
        'dataset': PermutedMNIST(10)
    }
    return kwargs

def splitmnist()->dict:
    kwargs = {
        'epochs': 20,
        'batch_size': 256,
        'coreset': 40,
        'model': SimpleMLP(num_classes=10, hidden_layers=2, hidden_size=256),#MTSimpleMLP(hidden_size=512),#SimpleMLP(num_classes=10, hidden_layers=2, hidden_size=256),SimpleMLP(num_classes=10, hidden_layers=2, hidden_size=256),#
        'vcl_model': VCLModel(num_classes=10, hidden_layers=2, hidden_size=256),
        'dataset': SplitMNIST(5)#, return_task_id=True
    }
    return kwargs

def splitcifar10()->dict:
    kwargs = {
        'epochs': 30,
        'batch_size': 256,
        'coreset': 40,
        'model': SimpleCNN(),#SimpleMLP(num_classes=10, hidden_layers=2, hidden_size=256),
        'dataset': SplitCIFAR10(5)#, return_task_id=True
    }
    return kwargs

def splitfmnist()->dict:
    kwargs = {
        'epochs': 20,
        'batch_size': 256,
        'coreset': 40,
        'model': SimpleMLP(num_classes=10, hidden_layers=4, hidden_size=200),
        'dataset': SplitFMNIST(5),
        'vcl_model': VCLModel(num_classes=10, hidden_layers=4, hidden_size=200),
    }
    return kwargs

def splitcifar100()->dict:
    raise NotImplementedError