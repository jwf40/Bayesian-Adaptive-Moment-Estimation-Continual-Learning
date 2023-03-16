import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from base import BaseCLMethod

class MAS(BaseCLMethod):
    def __init__(self, model, train_loader, test_loader, **kwargs):
        super().__init__(model, train_loader, test_loader, **kwargs)
        self.lambda_ = 1.0
        self.alpha_ = 0.5