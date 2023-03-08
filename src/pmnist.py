import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.sampler import RandomSampler
import numpy as np
from torchvision.datasets import MNIST
from torchvision.transforms import Compose
import random
from data import PermutedMNIST
from tqdm import tqdm
from copy import deepcopy
from optimizers_lib import bgd
from models.vcl_nn import MLP
from typing import Optional, Sized, Iterator

class RemoveAfterSampled(RandomSampler):
    """
    Used to draw samples and then remove them from the data
    When drawn, adds samples to a forbidden list
    Main datastructure can later be repopulated with said samples
    """    
    data_source: Sized
    forbidden: Optional[list]
    def __init__(self, data_source: Sized, forbidden: Optional[list] = [])-> None:
        super().__init__(data_source)
        self.forbidden = forbidden
        self.refill()
    
    def remove(self,new_forbidden):
        for num in new_forbidden:
            if not (num in self.forbidden):
                self.forbidden.append(num)
        self._remove(new_forbidden)

    def _remove(self, to_remove):
        for num in to_remove:
            if num in self.idx:
                self.idx.remove(num)
        self._num_samples = len(self.idx)
    
    def refill(self):
        # Refill the indices after iterating through the entire DataLoader
        self.idx = list(range(len(self.data_source)))
        self._remove(self.forbidden)

    def __iter__(self) -> Iterator[int]:
        for _ in range(self.num_samples // 32):
            batch = random.sample(self.idx, 32)
            self._remove(batch)

def get_pmnist_data(n_tasks, batch_size=128):
    train_loader = []
    test_loader = []
    idx = list(range(28 * 28))
    for i in range(n_tasks):
        train_loader.append(DataLoader(PermutedMNIST(train=True, permute_idx=idx, id=i), batch_size=batch_size,num_workers=4, shuffle=True))
        test_loader.append(DataLoader(PermutedMNIST(train=False, permute_idx=idx,  id=i),batch_size=batch_size))
        random.shuffle(idx)
    return (train_loader, test_loader)

def _train(model, dloader,head=0,device='cpu',epochs=100,lr=0.001):
    """
    Train on labelled data in a traditional point-wise manner
    """
    print("Beginning Training...")
    optimizer=optim.Adam(model.parameters(), lr=lr)
    for ep in tqdm(range(epochs)):
        for batch in dloader:
            optimizer.zero_grad()
            x,label = batch[0].to(device), batch[1].to(device)            
            loss = model.point_estimate_loss(x,label,head=head)
            loss.backward()
            optimizer.step()
        # if (ep % 10) == 0:
        #     _test(model, [dloader], device=device)
     

def _post_train(model, teacher,dloader,head=0,device='cpu',epochs=100,lr=0.001):
    """
    Train using VCL, in 
    """
    print("Going to Post-Train Mode")
    # device='cpu'
    # model = model.to(device)
    # teacher=teacher.to(device)
    optimizer=optim.Adam(model.parameters(), lr=lr)
    for _ in tqdm(range(epochs)):
        epoch_loss = 0
        for batch in dloader:
            optimizer.zero_grad()
            x, y= batch[0].to(device), batch[1].to(device)
            entropies = model.predictive_entropy_from_means(x)
            ood_data = torch.where(entropies>0.05)[0]
            if len(ood_data) > 0:
                teacher_means = teacher.get_prediction_means(x)

                better_perf_samples = torch.where(\
                    teacher.predictive_entropy_from_means(teacher_means[ood_data]) <0.1)

                smpl_idx = ood_data[better_perf_samples]

                teacher_preds = torch.argmax(teacher_means[smpl_idx], dim=1).to(device)
                loss = model.vcl_loss(x[smpl_idx], teacher_preds, head, len(teacher_preds))
                #epoch_loss += len(x) * loss.item()
                loss.backward()
                optimizer.step()
                model.reset_for_new_task(head) 


def _post_train_bgd(agents, dloaders,head=0,epochs=12,device='cpu',lr=0.001,batch_size=1):
    """
    Train using VCL, in 
    """
    print("Going to Post-Train Mode")
    # device='cpu'
    # model = model.to(device)
    # teacher=teacher.to(device)
    
    #Concatenate datasets, any sample can be selected
    dataset = ConcatDataset([ds.dataset for ds in dloaders])
    #
    dloader = DataLoader(dataset,batch_size=batch_size, shuffle=True)
    entropy_threshold = 0.001
    for epoch in tqdm(range(epochs*len(agents))):
        for batch in dloader:
            #Choose a random agent to train this batch
            model = agents[0]#random.choice(agents)
            #optimizer=bgd(model, mc_iters=None)
            optimizer=optim.Adam(model.parameters(), lr=lr)
            optimizer.zero_grad()
            #Calculate its entropy
            x, y= batch[0].to(device), batch[1].to(device)
            entropies = model.predictive_entropy_from_means(x)
            ood_data = torch.where(entropies>entropy_threshold)[0]
            #If we have uncertainty
            if len(ood_data) > 0:
                #Take samples forward and ask all other agents
                x = x[ood_data]
                y = y[ood_data]
                for idx, teacher in enumerate(agents):
                    if len(x)==0:
                        break
                    elif teacher != model:
                        #optimizer.randomize_weights()
                        teacher_means = teacher.get_prediction_means(x)
                        t_pred_ent = teacher.predictive_entropy_from_means(teacher_means)
                        better_perf_samples = torch.where(t_pred_ent<entropy_threshold)[0]
                        #If teacher knows an answer, update model
                        if len(better_perf_samples)>0:
                            smpl_idx = better_perf_samples
                            teacher_preds = torch.argmax(teacher_means[smpl_idx], dim=1).to(device)
                            loss = model.point_estimate_loss(x[smpl_idx],teacher_preds,head=head)
                            keep_idxs = np.delete(np.arange(len(x),step=1),better_perf_samples.detach().cpu())
                            x = x[keep_idxs]
                            y = y[keep_idxs]
                            #epoch_loss += len(x) * loss.item()
                            loss.backward()
                            #optimizer.aggregate_grads(len(smpl_idx))
                            optimizer.step()
                model.reset_for_new_task(head) 

def test_loop(agents, dloaders,test_data,head=0,epochs=12,device='cpu',lr=0.001,batch_size=1):
    """
    Train using VCL, in 
    """
    print("Going to Post-Train Mode")
    print(agents[0].entropy_threshold)
    # device='cpu'
    # model = model.to(device)
    # teacher=teacher.to(device)
    
    #Concatenate datasets, any sample can be selected
    dataset = ConcatDataset([ds.dataset for ds in dloaders])
    #
    dloader = DataLoader(dataset,batch_size=batch_size, shuffle=True)


    
    for epoch in tqdm(range(epochs*(len(agents)-1))):
        #used to choose agent below, too
        # idx = random.choice(range(len(dloaders)))
        # dl = dloaders[idx]
        for batch in tqdm(dloader):
            #Choose a random agent to train this batch
            model = random.choice(agents)
            model.train()
            optimizer=bgd(model, mc_iters=None)
            #optimizer=optim.Adam(model.parameters(), lr=lr)
            optimizer.zero_grad()
            #Calculate its entropy
            x, y, ds_ids= batch[0].to(device), batch[1].to(device), batch[2].to(device)
            entropies = model.predictive_entropy_from_means(x)
            ood_data = torch.where(entropies>model.entropy_threshold)[0]
            #If we have uncertainty
            if len(ood_data) > 0:
                #Take samples forward and ask all other agents
                for idx in set(ds_ids[ood_data]):
                    task_n_ood = ood_data[ds_ids[ood_data]==idx]
                    teacher = agents[idx]
                    if teacher != model:
                        teacher.eval()
                        optimizer.randomize_weights()
                        teacher_means = teacher.get_prediction_means(x)
                        t_pred_ent = teacher.predictive_entropy_from_means(teacher_means)
                        better_perf_samples = torch.where(t_pred_ent[task_n_ood]<teacher.entropy_threshold)[0]
                        #If teacher knows an answer, update model
                        if len(better_perf_samples)>0:
                            smpl_idx = task_n_ood[better_perf_samples]
                            teacher_preds = torch.argmax(teacher_means[smpl_idx], dim=1).to(device)
                            #loss = model.vcl_loss(x[smpl_idx], teacher_preds, head, len(teacher_preds))
                            a = y[smpl_idx]
                            loss = model.point_estimate_loss(x[smpl_idx],teacher_preds,head=head)
                            loss.backward()
                            optimizer.aggregate_grads(len(smpl_idx))
                            optimizer.step()
    #model.reset_for_new_task(head) 


def _post_train_ask_all_agents(agents, dloaders,head=0,epochs=12,device='cpu',lr=0.001,batch_size=1):
    """
    Train using VCL, in 
    """
    print("Going to Post-Train Mode")
    # device='cpu'
    # model = model.to(device)
    # teacher=teacher.to(device)
    
    #Concatenate datasets, any sample can be selected
    dataset = ConcatDataset([ds.dataset for ds in dloaders])
    #
    dloader = DataLoader(dataset,batch_size=batch_size)
        
    for epoch in tqdm(range(epochs*len(agents))):
        for batch in dloader:
            #Choose a random agent to train this batch
            model = random.choice(agents)
            optimizer=optim.Adam(model.parameters(), lr=lr)
            optimizer.zero_grad()
            #Calculate its entropy
            x, y= batch[0].to(device), batch[1].to(device)
            entropies = model.predictive_entropy_from_means(x)
            ood_data = torch.where(entropies>0.05)[0]
            #If we have uncertainty
            if len(ood_data) > 0:
                #Take samples forward and ask all other agents
                x = x[ood_data]
                for idx, teacher in enumerate(agents):
                    if len(x)==0:
                        break
                    if teacher != model:
                        teacher_means = teacher.get_prediction_means(x)
                        t_pred_ent = teacher.predictive_entropy_from_means(teacher_means)
                        better_perf_samples = torch.where(t_pred_ent<0.05)[0]
                        #If teacher knows an answer, update model
                        if len(better_perf_samples)>0:
                            smpl_idx = better_perf_samples
                            teacher_preds = torch.argmax(teacher_means[smpl_idx], dim=1).to(device)
                            loss = model.vcl_loss(x[smpl_idx], teacher_preds, head, len(teacher_preds))
                            keep_idxs = np.delete(np.arange(len(x),step=1),better_perf_samples.detach().cpu())
                            x = x[keep_idxs]
                #epoch_loss += len(x) * loss.item()
                loss.backward()
                optimizer.step()
                model.reset_for_new_task(head) 


def _test(model,dloader_li,head=0,device='cpu'):
    print("Starting Testing")
    task_accuracies =[]
    task_entropies = []
    for task_data in tqdm(dloader_li):
        acc = 0
        avg_entropy = torch.Tensor([]).to(device)
        for data in tqdm(task_data):
            x,y = data[0].to(device), data[1].to(device)
            y_preds = model.prediction(x, head).to(device)
            acc += (y_preds.int() == y.int()).sum().item()/len(y)
            entropy = model.predictive_entropy(x[y_preds.int() == y.int()])
            avg_entropy = torch.concat([avg_entropy,entropy])
        task_accuracies.append(100*acc/len(task_data))
        task_entropies.append(avg_entropy.median().item())
        model.entropy_threshold = min(task_entropies)
    print(f"Task Accuracies: {task_accuracies}")
    print(f"Task Entropies: {task_entropies}")

def pmnist_exp(n_tasks=3, device='cpu'):
    N_CLASSES = 10
    INPUT_SHAPE = 28*28
    LAYER_WIDTH = 100
    N_HIDDEN_LAYERS = 2
    N_TASKS = n_tasks
    EPOCHS = 1
    BATCH_SIZE = 128
    INITIAL_POSTERIOR_VAR = 1e-3
    #create 
    agents = [
        MLP(
            in_size=INPUT_SHAPE, out_size=N_CLASSES,
            layer_width=LAYER_WIDTH, n_hidden_layers=N_HIDDEN_LAYERS,
            n_heads=1,
            initial_posterior_var=INITIAL_POSTERIOR_VAR
        ).to(device) \
            for _ in range(n_tasks) 
    ]

    train_loader, test_loader = get_pmnist_data(n_tasks=N_TASKS,batch_size=BATCH_SIZE)

    for idx, (agent,trldr) in enumerate(zip(agents, train_loader)):
        print(f"Agent {idx}...")
        agent.id = idx
        _train(agent,trldr, device=device, epochs=EPOCHS)
        _test(agent, test_loader,device=device)
    
    test_loop(agents,train_loader,test_data=test_loader, batch_size=128, device=device, epochs=1)
    #_post_train_bgd(agents,train_loader, batch_size=128, device=device, epochs=EPOCHS)
    
    for idx, agent in enumerate(agents):
        print(f"Agent {idx}...")
        print(agent.entropy_threshold)
        _test(agent, test_loader, device=device)
    