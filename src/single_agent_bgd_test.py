import numpy as np
import random
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, ConcatDataset, Subset
import torchvision
from torchvision.datasets import MNIST
from torchvision.transforms import Compose
import torchvision.transforms as transforms

from tqdm import tqdm
from data import PermutedMNIST
from models.basic_mlp import BasicMLP
from optimizers_lib import fastbgd, bgd


def get_pmnist_data(n_tasks, batch_size=128):    
    train_loader = []
    test_loader = []

    idx = list(range(28 * 28))
    for i in range(n_tasks):
        random.shuffle(idx)
        train_loader.append(DataLoader(PermutedMNIST(train=True, permute_idx=idx, id=i), batch_size=batch_size,num_workers=1, shuffle=True))
        test_loader.append(DataLoader(PermutedMNIST(train=False, permute_idx=idx,  id=i),batch_size=batch_size))
        
    return (train_loader, test_loader)


def train_bgd(model, dloaders, test_loaders,epochs,device='cpu', fast=False, mean_eta=1):
    opt = bgd(model, mean_eta=mean_eta, std_init=0.06) if not fast else fastbgd(model, mean_eta=mean_eta, std_init=0.06)
    criterion = nn.CrossEntropyLoss()
    running_accs = []
    running_cost = [[] for _ in range(len(dloaders))]
    for dl_idx, dl in enumerate(tqdm(dloaders)):
        for epoch in tqdm(range(epochs)):
            rloss = 0
            for b_idx,batch in enumerate(dl):
                x, y = batch[0].to(device), batch[1].to(device)
                mcloss = 0
                for mc_iter in range(opt.mc_iters):
                    opt.randomize_weights()
                    output = model(x)
                    loss = criterion(output, y)
                    mcloss += loss
                    #print(loss)
                    opt.zero_grad()
                    loss.backward()
                    opt.aggregate_grads(len(y))
                mcloss/=opt.mc_iters
                opt.step()
                # if b_idx%20==0:
                #     running_accs.append(test(model,test_loaders,device))  
                #     print(running_accs[-1])
            running_cost[dl_idx].append(rloss/len(dl))
            if dl_idx>0:
                running_accs.append(np.mean(test(model,test_loaders,device)[:dl_idx]))
            else: 
                running_accs.append(np.mean(test(model,test_loaders,device)[0]))
    print(running_accs[-1])
    with open(f'Results/Mean_Eta_{mean_eta}_Fast_{fast}_bgd_single_agent', 'wb') as f:
        pickle.dump(running_accs, f)    
    with open(f'Results/LOSS_Mean_Eta_{mean_eta}_Fast_{fast}_bgd_single_agent', 'wb') as f:
        pickle.dump(running_cost, f)    
        
def train_sgd(model, dloaders, test_loaders,epochs,device='cpu', adam=False):
    opt = optim.SGD(model.parameters(), lr=0.01) if not adam else optim.Adam(model.parameters(), lr=0.001,betas=(0.9,0.999))
    criterion = nn.CrossEntropyLoss()
    running_accs = []
    running_cost = []
    for dl_idx,dl in enumerate(tqdm(dloaders)):
        for epoch in tqdm(range(epochs)):
            for b_idx,batch in enumerate(dl):
                x, y = batch[0].to(device), batch[1].to(device)
                output = model(x)
                loss = criterion(output, y)
                running_cost.append(loss/len(y))
                loss.backward()
                opt.step()
            if dl_idx>0:
                running_accs.append(np.mean(test(model,test_loaders,device)[:dl_idx]))
            else: 
                running_accs.append(np.mean(test(model,test_loaders,device)[0]))
        #test(model,test_loaders,device)
    print(running_accs[-1])
    with open(f'Results/Adam_{adam}_sgd_single_agent', 'wb') as f:
        pickle.dump(running_accs, f)

        
def test(model, dloaders, device='cpu'):
    accs = []
    for idx,dl in enumerate(dloaders):
        running_acc = 0
        for batch in dl:
            x, y = batch[0].to(device), batch[1].to(device)
            preds = model(x)
            running_acc += (torch.argmax(preds, dim=1)==y).sum()/len(y)
        acc = running_acc/len(dl)
        accs.append(acc.item())
    print(f'Task acc: {accs}')
    return accs

if __name__=='__main__':
    N_TASKS = 10
    BATCH_SIZE = 128
    EPOCHS = 1
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # random.seed(12345)
    # np.random.seed(12345)
    # torch.manual_seed(12345)

    train_loader, test_loader = get_pmnist_data(n_tasks=N_TASKS,batch_size=BATCH_SIZE)

    model = BasicMLP().to(DEVICE)    
    train_sgd(model,train_loader,test_loader,EPOCHS,DEVICE, adam=True)
    # model = BasicMLP().to(DEVICE)
    # train_sgd(model,train_loader,test_loader,EPOCHS,DEVICE, adam=False)
    # model = BasicMLP().to(DEVICE)
    # train_bgd(model,train_loader,test_loader,EPOCHS,DEVICE, fast=False, mean_eta=10)
    # model = BasicMLP().to(DEVICE)
    # train_bgd(model,train_loader,test_loader,EPOCHS,DEVICE, fast=False, mean_eta=1)
    # model = BasicMLP().to(DEVICE)
    # train_bgd(model,train_loader,test_loader,EPOCHS,DEVICE, fast=True, mean_eta=0.1)
