import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import random
from torch.utils.data import DataLoader
from data import GraduatedDataLoader
from torchvision.transforms import (
    ToTensor,
    ToPILImage,
    Compose,
    Normalize,
    RandomRotation,
)
from avalanche.benchmarks.classic import PermutedMNIST
from avalanche.benchmarks.datasets.external_datasets.mnist import \
    get_mnist_dataset
from avalanche.models import SimpleMLP
from avalanche.training import EWC
from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.utils import make_avalanche_dataset, make_classification_dataset
from PIL.Image import Image
from torch import Tensor




# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model
model = SimpleMLP(num_classes=10,hidden_size=1000,
        hidden_layers=2,
        drop_rate=0.0)


benchmark = PermutedMNIST(n_experiences=10)

train_stream = benchmark.train_stream
test_stream = benchmark.test_stream


# Prepare for training & testing
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = CrossEntropyLoss()

# Continual learning strategy
cl_strategy = EWC(
    model, optimizer, criterion, ewc_lambda=1000,train_mb_size=128, train_epochs=10,
    eval_mb_size=128, device=device)

# train and test loop over the stream of experiences
results = []
for train_exp in train_stream:
    print(train_exp)
    cl_strategy.train(train_exp)
    results.append(cl_strategy.eval(test_stream))