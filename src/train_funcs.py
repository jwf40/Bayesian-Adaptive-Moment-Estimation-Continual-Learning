import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from optimizers_lib import fastbgd, bgd


def train_bgd(model, dloaders, test_loaders,epochs,device='cpu', fast=False, mean_eta=1.0):
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
