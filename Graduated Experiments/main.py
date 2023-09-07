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
parser.add_argument('-a','--alg', help='continual learning algorithm to use',choices=['badam', 'tfcl','bgd','naive', 'ewconline','mas','synaptic_intelligence', 'vcl'],required=True)
parser.add_argument('-x','--exp', help='experiment to run, and dataset to use.',choices=['CIsplitmnist','CIsplitfmnist','pmnist'], required=True)
parser.add_argument('-g', '--graduated', type=bool,help='whether or not to use graduated data', required=False, default=True)
parser.add_argument('-e','--epochs', type=int,help='number of epochs per task', required=False, default=1)
parser.add_argument('-b','--batch_size', type=int, help='batch size', required=False, default=128)
parser.add_argument('-d', '--device', help='device to run on', required=False, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('-k', '--grad_k', type=int, help='exponent of equation to dictate graduation in data', required=False, default=2)
parser.add_argument('-r', '--root', type=str, help='Root folder to save runs', required=False, default='')
args = vars(parser.parse_args())
print(args)

if __name__=='__main__':    
    for run in range(1,26):
        print("WANDB is in base.py def run()")
        
        random.seed(run)
        np.random.seed(run)
        torch.manual_seed(run)
        test_every = 40 if 'pmnist' not in args['exp'] else 400
        n_task = 10 if args['exp'] == 'pmnist' else 5                        
        print(f"Starting training of {args['alg']} on  {args['exp']}")
        args = {'run': run, 'test_every': test_every, 'shuffle': True, 'buffer_len':1,'run': run,'alg': args['alg'], 'exp': args['exp'], 'graduated': True, 'k': 2,'epochs': 1, 'batch_size': 128, 'labels':False, 'n_tasks': n_task, 'device': 'cuda', 'root': 'results/test_acc/Final_Shuffle_NoLabels/'}#
        getattr(experiments, f"{args['alg']}_main")(**args)
            
            
    