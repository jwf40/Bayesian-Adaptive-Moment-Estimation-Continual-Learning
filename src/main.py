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



# parser = argparse.ArgumentParser(description='Continual Learning Script')
# parser.add_argument('-a','--alg', help='continual learning algorithm to use', required=True)
# parser.add_argument('-x','--exp', help='experiment to run, and dataset to use', required=True)
# parser.add_argument('-g', '--graduated', type=bool,help='whether or not to use graduated data', required=False, default=False)
# parser.add_argument('-e','--epochs', type=int,help='number of epochs per task', required=False, default=1)
# parser.add_argument('-b','--batch_size', type=int, help='batch size', required=False)
# parser.add_argument('-t','--n_tasks', type=int, help='number of tasks', required=False)
# parser.add_argument('-d', '--device', help='device to run on', required=False, default='cuda' if torch.cuda.is_available() else 'cpu')
# parser.add_argument('-l', '--labels', type=bool, help='Whether task labels are available to method', required=False, default=False)
# parser.add_argument('-k', '--grad_k', type=int, help='exponent of equation to dictate graduation in data', required=False, default=12)
# parser.add_argument('-r', '--root', type=str, help='Root folder to save runs', required=False, default='results/test_acc/')
# args = vars(parser.parse_args())
# print(args)

if __name__=='__main__':    
    exps = ['CIsplitmnist']#,,, ,'pmnist', 'CIcifar','DIcifar',,'pmnist','DIsplitmnist',]
    algs = ['bgd']#, #,'ewconline','tfcl','vcl','mas','synaptic_intelligence',  ,'bufferbadam',,'bgd','ewconline','tfcl','mas','synaptic_intelligence', 'vcl''synaptic_intelligence']#'ewc',
    for exp in exps:
        for alg in algs:
            for run in range(10):                                   
                random.seed(run)
                np.random.seed(run)
                torch.manual_seed(run)
                test_every = 40 if 'pmnist' not in exp else 400
                n_task = 10 if exp == 'pmnist' else 5                        
                print(f"Starting training of {alg} on  {exp}")
                args = {'test_every': test_every, 'shuffle': True, 'buffer_len':1,'run': run,'alg': alg, 'exp': exp, 'graduated': False, 'k': 2,'epochs': 20, 'batch_size': 128, 'n_tasks': n_task, 'device': 'cuda', 'labels': True, 'root': 'results/test_acc/Labelled_Final/'}#
                getattr(experiments, f"{args['alg']}_main")(**args)
    
    # exps = ['CIsplitmnist']#,,, ,'pmnist', 'CIcifar','DIcifar',,'pmnist','DIsplitmnist',]
    # algs = ['bgd']#, #,'ewconline','tfcl','vcl','mas','synaptic_intelligence',  ,'bufferbadam',,'bgd','ewconline','tfcl','mas','synaptic_intelligence', 'vcl''synaptic_intelligence']#'ewc',
    # stds = [0.001, 0.002, 0.003, 0.004,0.005,0.006]
    # etas = [0.5, 1.0, 5]
    # for exp in exps:
    #     for alg in algs:
    #         for std in stds:
    #             for eta in etas:
    #                 for run in range(10):                
    #                     random.seed(run)
    #                     np.random.seed(run)
    #                     torch.manual_seed(run)
    #                     test_every = 40 if 'pmnist' not in exp else 400
    #                     n_task = 10 if exp == 'pmnist' else 5                        
    #                     print(f"Starting training of {alg} on  {exp}")
    #                     args = {'new_std': std, 'new_eta': eta,'test_every': test_every, 'shuffle': True, 'buffer_len':1,'run': run,'alg': alg, 'exp': exp, 'graduated': False, 'k': 2,'epochs': 20, 'batch_size': 256, 'n_tasks': n_task, 'device': 'cuda', 'labels': True, 'root': 'results/test_acc/Labelled_Final/'}#
    #                     getattr(experiments, f"{args['alg']}_main")(**args)
    
    # for exp in exps:
    #     for alg in algs:
    #         print(f"Starting training of {alg} on  {exp}")
    #         args = {'alg': alg, 'exp': exp, 'graduated': False, 'k': 12,'epochs': 10, 'batch_size': 128, 'n_tasks': 5, 'device': 'cuda', 'labels': True, 'root': 'results/test_acc/'}
    #         getattr(experiments, f"{args['alg']}_main")(**args)
            
            
    