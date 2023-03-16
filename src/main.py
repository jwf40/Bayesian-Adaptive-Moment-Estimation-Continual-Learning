import numpy as np
import random
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision.datasets import MNIST
from torchvision.transforms import Compose
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm

from utils import split_args
import experiments
from optimizers_lib import fastbgd, bgd



parser = argparse.ArgumentParser(description='Continual Learning Script')
parser.add_argument('-a','--alg', help='continual learning algorithm to use', required=True)
parser.add_argument('-x','--exp', help='experiment to run, and dataset to use', required=True)
parser.add_argument('-g', '--graduated', type=bool,help='whether or not to use graduated data', required=False, default=False)
parser.add_argument('-e','--epochs', type=int,help='number of epochs per task', required=False, default=1)
parser.add_argument('-b','--batch_size', type=int, help='batch size', required=False)
parser.add_argument('-t','--n_tasks', type=int, help='number of tasks', required=False)
parser.add_argument('-d', '--device', help='device to run on', required=False, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('-l', '--labels', type=bool, help='Whether task labels are available to method', required=True)
args = vars(parser.parse_args())
print(args)

if __name__=='__main__':
    # random.seed(12345)
    # np.random.seed(12345)
    # torch.manual_seed(12345)

   
    getattr(experiments, f"{args['alg']}_main")(**args)
    
    #train_sgd(model,train_loader,test_loader,EPOCHS,DEVICE, adam=True)
    # model = BasicMLP().to(DEVICE)
    # train_sgd(model,train_loader,test_loader,EPOCHS,DEVICE, adam=False)
    # model = BasicMLP().to(DEVICE)
    #train_bgd(model,train_loader,test_loader,EPOCHS,DEVICE, fast=False, mean_eta=1)
    # model = BasicMLP().to(DEVICE)
    # train_bgd(model,train_loader,test_loader,EPOCHS,DEVICE, fast=False, mean_eta=1)
    # model = BasicMLP().to(DEVICE)
    #train_bgd(model,train_loader,test_loader,EPOCHS,DEVICE, fast=True, mean_eta=0.1)
