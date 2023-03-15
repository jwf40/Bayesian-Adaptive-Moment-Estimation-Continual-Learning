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
import data
import experiments
from data import GraduatedDataLoader
from models.basic_mlp import BasicMLP
from optimizers_lib import fastbgd, bgd



parser = argparse.ArgumentParser(description='Continual Learning Script')
parser.add_argument('-a','--alg', help='continual learning algorithm to use', required=True)
parser.add_argument('-d','--ds', help='dataset to use', required=True)
parser.add_argument('-g', '--graduated', type=bool,help='whether or not to use graduated data', required=False, default=False)
parser.add_argument('-e','--epochs', type=int,help='number of epochs per task', required=False)
parser.add_argument('-b','--batch_size', type=int, help='batch size', required=False)
parser.add_argument('-t','--n_tasks', type=int, help='number of tasks', required=False)
args = vars(parser.parse_args())

if __name__=='__main__':
    N_TASKS = 5
    BATCH_SIZE = 128
    EPOCHS = 3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # random.seed(12345)
    # np.random.seed(12345)
    # torch.manual_seed(12345)

    # These args are only needed here, don't want to pass them forward
    #launch_kwargs, kwargs = split_args(args)

    # Get requested dataset 
    train_loader, test_loader,dataset_kwargs = getattr(data, f'get_{args["ds"]}')(**args)#get_pmnist_data(n_tasks=N_TASKS,batch_size=BATCH_SIZE)
    print(dataset_kwargs)
    if args['graduated']:
        train_loader = GraduatedDataLoader(train_loader)
    
    getattr(experiments, f"{args['alg']}_main")(**args, **dataset_kwargs)
    
    #train_sgd(model,train_loader,test_loader,EPOCHS,DEVICE, adam=True)
    # model = BasicMLP().to(DEVICE)
    # train_sgd(model,train_loader,test_loader,EPOCHS,DEVICE, adam=False)
    # model = BasicMLP().to(DEVICE)
    #train_bgd(model,train_loader,test_loader,EPOCHS,DEVICE, fast=False, mean_eta=1)
    # model = BasicMLP().to(DEVICE)
    # train_bgd(model,train_loader,test_loader,EPOCHS,DEVICE, fast=False, mean_eta=1)
    # model = BasicMLP().to(DEVICE)
    #train_bgd(model,train_loader,test_loader,EPOCHS,DEVICE, fast=True, mean_eta=0.1)
