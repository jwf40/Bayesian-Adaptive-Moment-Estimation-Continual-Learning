import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.utils.data.sampler import RandomSampler
import numpy as np
from torchvision.datasets import MNIST
from torchvision.transforms import Compose
import random
from data import PermutedMNIST
from tqdm import tqdm
from copy import deepcopy
from models.bnn import MLP
from optimizers_lib import BGD, bgd
import pickle
from typing import Optional, Sized, Iterator

def get_pmnist_data(n_tasks, batch_size=128):
    train_loader = []
    test_loader = []
    idx = list(range(28 * 28))
    for i in range(n_tasks):
        ds = PermutedMNIST(train=True, permute_idx=idx, id=i)
        train_loader.append(DataLoader(Subset(ds,np.linspace(0,len(ds)-1,len(ds), dtype=int)), batch_size=batch_size,num_workers=1, shuffle=True))
        test_loader.append(DataLoader(PermutedMNIST(train=False, permute_idx=idx,  id=i),batch_size=batch_size))
        random.shuffle(idx)
    return (train_loader, test_loader)

def _train(model, dloader,device='cpu',epochs=100,lr=0.001):
    """
    Train on labelled data in a traditional point-wise manner
    """
    print("Beginning Training...")
    optimizer=optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for ep in tqdm(range(epochs)):
        for batch in dloader:
            optimizer.zero_grad()
            x,label = batch[0].to(device), batch[1].to(device)            
            prediction = model(x)
            loss = criterion(prediction, label)
            loss.backward()
            optimizer.step()

def _test(model,dloader_li,device='cpu'):
    print("Starting Testing")
    task_accuracies =[]
    task_entropies = []
    for task_data in tqdm(dloader_li):
        acc = 0
        avg_entropy = torch.Tensor([]).to(device)
        for data in tqdm(task_data):
            x,y = data[0].to(device), data[1].to(device)
            y_preds = torch.argmax(model(x), dim=1)
            acc += (y_preds.int() == y.int()).sum().item()/len(y)
            entropy = model.predictive_entropy(x)#[y_preds.int() == y.int()]
            avg_entropy = torch.concat([avg_entropy,entropy])
        avg_entropy = avg_entropy.detach().cpu()
        task_accuracies.append(100*acc/len(task_data))
        task_entropies.append(avg_entropy.mean().item()+(2*avg_entropy.std().item()))
        #task_entropies.append(np.quantile(avg_entropy.detach().cpu(), 0.5))
#        task_entropies.append(avg_entropy.mean().item())
        model.entropy_threshold = min(task_entropies)
    with open(f'Results/pmnist/trained_accuracies_agent_{model.id}.pickle', 'wb') as f:
        pickle.dump(task_accuracies, f)
    with open(f'Results/pmnist/trained_entropies_agent_{model.id}.pickle', 'wb') as f:
        pickle.dump(task_entropies, f)
    print(f"Task Accuracies: {task_accuracies}")
    print(f"Mean Accuracy {sum(task_accuracies)/len(task_accuracies)}")
    print(f"Task Entropies: {task_entropies}")


def test_collab(agents,dloaders,device='cpu'):
    print("Testing Collab...")
    task_accs = [0 for _ in range(len(dloaders))]
    acc_matrix = [[0,0] for _ in range(len(dloaders))]
    model = agents[0]
    for idx, dloader in enumerate(dloaders):
        for batch in tqdm(dloader):
            x,y = batch[0].to(device), batch[1].to(device)
            #model = random.choice(agents)

            model_preds = model.prediction_means(x)
            model_entropy = model.predictive_entropy_from_means(model_preds)

            low_certainty_samples = torch.where(model_entropy>model.entropy_threshold)[0]
            high_certainty_samples = np.delete(np.arange(len(x),step=1),low_certainty_samples.detach().cpu())
            task_accs[idx] += ((len(high_certainty_samples)*int(model.id==idx))\
                                + len(low_certainty_samples)*int(model.id!=idx)
                                )
            acc_matrix[idx][0] += len(high_certainty_samples)*int(model.id==idx)
            acc_matrix[idx][1] += len(low_certainty_samples)*int(model.id!=idx)
    
    for i in range(len(dloaders)):
        task_accs[i]  /= len(dloaders[i].dataset)
        acc_matrix[i][0] /= len(dloaders[i].dataset)
        acc_matrix[i][1] /= len(dloaders[i].dataset)
    print(task_accs)
    print(acc_matrix)
    with open('./Results/pmnist/collab_accuracy.pickle', 'wb') as f:
        pickle.dump(task_accs, f)
    with open('./Results/pmnist/collab_accuracy_matrix.pickle', 'wb') as f:
        pickle.dump(acc_matrix, f)

# def train_matchmaking_head(model, agents, optims,x, past_agents,true_label=None):
#     """
#     Recursive Function to find viable teaching candidate
#     """

#     optimizer = optims[model.id]
#     optimizer.zero_grad()
#     #Can't ask agents twice
#     past_agents.append(model.id)

#     #Base case, If we have no more teachers
#     if len(past_agents)>=len(agents):
#         return None

#     #Choose a teacher
#     conf_in_teachers= model.choose_teacher(x).mean(dim=0)
    
#     #Sort teacher idxs by confidence in them
#     rank_choices = sorted(range(len(conf_in_teachers)), key=lambda x: conf_in_teachers[x])
    
#     #Remove previously asked agents
#     for idx in past_agents:
#         rank_choices.remove(idx)

#     #Select highest confidence agent that remains
#     teacher_choice = rank_choices[-1]
    
#     #Assign teacher
#     teacher = agents[teacher_choice]
#     #Get predictions and entropies
#     teacher_preds = teacher.prediction_means(x)
#     teacher_entropy = teacher.predictive_entropy_from_means(teacher_preds)
#     #If teacher was correctly chosen
#     if round(teacher.is_confident(teacher_entropy).float().mean(0).item()):
#         #train
#         label = torch.zeros_like(conf_in_teachers).to(device)
#         label[teacher_choice] = 1.0
#         loss = nn.CrossEntropyLoss()(conf_in_teachers,label)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         return teacher_choice
#     else:        
#         #Pass task on to next agent
#         next_choice = train_matchmaking_head(teacher, agents, optims, x, past_agents,true_label=true_label) 
#         #If we find an agent, propagate the label backwards
#         if next_choice:
#             #And train
#             label = torch.zeros_like(conf_in_teachers).to(device)
#             label[next_choice] = 1.0
#             loss = nn.CrossEntropyLoss()(conf_in_teachers,label)
#             loss.backward()
#             optimizer.step()
#         return next_choice
        

def train_matchmaking_head(model, agents, optims,x, past_agents,true_label=None):
    """
    Recursive Function to find viable teaching candidate
    """
    optimizer = optims[model.id]
    #optimizer.randomize_weights()
    optimizer.zero_grad()
    #Can't ask agents twice
    past_agents.append(model.id)

    #Choose a teacher
    conf_in_teachers= model.choose_teacher(x).mean(dim=0)
    
    #Sort teacher idxs by confidence in them
    rank_choices = sorted(range(len(conf_in_teachers)), key=lambda x: conf_in_teachers[x])
    
    #Remove previously asked agents
    for idx in past_agents:
        rank_choices.remove(idx)

    #Select highest confidence agent that remains
    teacher_choice = rank_choices[-1]
    
    #Assign teacher
    teacher = agents[teacher_choice]
    #Get predictions and entropies
    teacher_preds = teacher.prediction_means(x)
    teacher_entropy = teacher.predictive_entropy_from_means(teacher_preds)
    task_label = torch.argmax(teacher_preds, dim=1).to(device)
    #If teacher was correctly chosen
    if round(teacher.is_confident(teacher_entropy).float().mean(0).item()):
        #train mm head
        label = torch.zeros_like(conf_in_teachers).to(device)
        label[teacher_choice] = 1.0
        loss = nn.CrossEntropyLoss()(conf_in_teachers,torch.tensor(teacher_choice).to(device))
        loss.backward()
        optimizer.aggregate_grads(x.size()[0])
        optimizer.step()
        optimizer.zero_grad()
       
        return teacher_choice,task_label
    else:   
        #Base case, If we have no more teachers
        #Must be here, if its above it can break!
        if len(past_agents)>=len(agents):
            return None,None
        #Pass task on to next agent
        next_choice,task_label = train_matchmaking_head(teacher, agents, optims, x, past_agents,true_label=true_label) 
        #If we find an agent, propagate the label backwards
        if next_choice != None:
            #And train
            label = torch.zeros_like(conf_in_teachers).to(device)
            label[next_choice] = 1.0
            loss = nn.CrossEntropyLoss()(conf_in_teachers,torch.tensor(next_choice).to(device))
            loss.backward()
            optimizer.aggregate_grads(x.size()[0])
            optimizer.step()
            optimizer.zero_grad()
        return next_choice,task_label


       
def post_train(agents,dloaders,test_loader=None,epochs=1,device='cpu'):
    #On average we will ask everyone
    epochs_li = range(int((epochs)*len(dloaders[0])*len(dloaders)))
    choice_li = [0 for _ in range(len(dloaders))]
    agent_choice = [0 for _ in range(len(agents))]
    test_matchmaking(agents,test_loader,device=device)
    running_test_acc = []
    optims = [BGD(model.matchmaking_layers(), std_init=0.02,mean_eta=1,mc_iters=1) for model in agents]
    task_optims = [BGD(model.prediction_layers(), std_init=0.02, mean_eta=1,mc_iters=1) for model in agents]
        #Normally top layer
    iter_loaders = [iter(dl) for dl in dloaders]
    for epoch in tqdm(epochs_li):    
        #print(f'Label Choice Accuracy: {choice_accs}')
        for optimizer,task_optim in zip(optims,task_optims):
            optimizer.zero_grad()
            task_optim.zero_grad()
            #task_optim.randomize_weights()
        
        idx = random.choice(range(len(dloaders)))            
        choice_li[idx] += 1
        batch = next(iter_loaders[idx])

        x,y = batch[0].to(device),batch[1].to(device)
        
        model = random.choice(agents)
        while model.id == idx:
            model = random.choice(agents)

        task_optim = task_optims[model.id]

        agent_choice[model.id] += 1
        model_preds = model.prediction_means(x)
        model_entropy = model.predictive_entropy_from_means(model_preds)
        if not round(model.is_confident(model_entropy).float().mean(0).item()):
            _,label = train_matchmaking_head(model,agents,optims,x, [],true_label=idx)
            #train prediction
            loss = nn.CrossEntropyLoss()(model.prediction_means(x), label)
            loss.backward()
            task_optim.aggregate_grads(x.size()[0])
            task_optim.step()
            task_optim.zero_grad()
        if test_loader and (epoch % 10 == 0):
            print("Evaluating Performance on Train Set...")
            print(f"Choice Distribution: {choice_li}")
            print(f"Agent Distribution: {agent_choice}")
            running_test_acc.append(sum(test_matchmaking(agents,test_loader,device=device)))
            with open('Results/pmnist/all_agents_decentralised_matchmaking_head_running_acc','wb') as f:
                pickle.dump(running_test_acc, f)
            for each in agents:
                _test(each,test_loader,device=device)
            
def sequential_post_train_one_agent(agents,dloaders,test_loader=None,epochs=1,device='cpu'):
    #On average we will ask everyone
    epochs_li = range(int((epochs)*len(dloaders[0])*len(dloaders)))
    choice_li = [0 for _ in range(len(dloaders))]
    agent_choice = [0 for _ in range(len(agents))]
    test_matchmaking(agents,test_loader,device=device)
    running_test_acc = []
    optims = [BGD(model.matchmaking_layers(), std_init=0.02,mean_eta=1,mc_iters=1) for model in agents]
    task_optims = [BGD(model.prediction_layers(), std_init=0.02, mean_eta=1,mc_iters=1) for model in agents]
    model = random.choice(agents)
    for idx in range(len(dloaders)):
        if model.id == idx:
            continue
        for batch_epoch,batch in enumerate(dloaders[idx]):
            for optimizer,task_optim in zip(optims,task_optims):
                optimizer.zero_grad()
                task_optim.zero_grad()
                #task_optim.randomize_weights()
            
            choice_li[idx] += 1
            x,y = batch[0].to(device),batch[1].to(device)
            
            task_optim = task_optims[model.id]

            agent_choice[model.id] += 1
            model_preds = model.prediction_means(x)
            model_entropy = model.predictive_entropy_from_means(model_preds)
            if not round(model.is_confident(model_entropy).float().mean(0).item()):
                _,label = train_matchmaking_head(model,agents,optims,x, [],true_label=idx)
                #train prediction
                loss = nn.CrossEntropyLoss()(model.prediction_means(x), label)
                loss.backward()
                task_optim.aggregate_grads(x.size()[0])
                task_optim.step()
                task_optim.zero_grad()
            if test_loader and (batch_epoch % 50 == 0):
                print("Evaluating Performance on Train Set...")
                print(f"Choice Distribution: {choice_li}")
                print(f"Agent Distribution: {agent_choice}")
                running_test_acc.append(sum(test_matchmaking(agents,test_loader,device=device)))
                with open('Results/pmnist/single_agent_sequential_train_acc','wb') as f:
                    pickle.dump(running_test_acc, f)
                for each in agents:
                    _test(each,test_loader,device=device)

def test_matchmaking(agents,dloaders,device='cpu'):
    #Ensure batch_size is 1
    # for idx,each in enumerate(dloaders):
    #     if each.batch_size != 1:
    #         dloaders[idx] = DataLoader(each.dataset, batch_size=1)
    task_accuracies = []
    agent_choices = [0 for _ in range(len(agents))]

    for idx,task_data in enumerate(tqdm(dloaders)):
        acc = 0
        for data in tqdm(task_data):
            model = random.choice(agents)            
            while(model.id==idx):
                model = random.choice(agents)
            agent_choices[model.id] += 1

            x = data[0].to(device)            
            y_preds = torch.argmax(model.choose_teacher(x), dim=1)            
            acc += (y_preds.int() == idx).sum().item()/len(y_preds)

        task_accuracies.append(100*acc/len(task_data))
        
    print(f"Matchmaking Accuracy: {task_accuracies}")
    print(f"Test Agent Choices: {agent_choices}")
    return task_accuracies



def test_with_matchmaking(agents,dloaders,epochs=1,batch_size=1,device='cpu'):
    print("testing with matchmaking head..")
    task_accs = [0 for _ in range(len(dloaders))]
    for idx, dloader in enumerate(dloaders):
        for iter,batch in enumerate(tqdm(dloader)):
            x,y = batch[0].to(device), batch[1].to(device)
            
            model = random.choice(agents)

            model_preds = model.prediction_means(x)
            model_entropy = model.predictive_entropy_from_means(model_preds)

            low_certainty_samples = torch.where(model_entropy>model.entropy_threshold)[0]
            if low_certainty_samples.numel() and model != agents[idx]:
                teacher = agents[idx]
                teacher_preds = teacher.prediction_means(x)
                teacher_entropy = teacher.predictive_entropy_from_means(teacher_preds)
                teachable_samples = torch.where(teacher_entropy<teacher.entropy_threshold)[0]
                non_teachable_samples = np.delete(np.arange(len(x),step=1),teachable_samples.detach().cpu())
                #TODO add loss here
                assert len(teachable_samples)+len(non_teachable_samples) == len(y)
                if len(teachable_samples):
                    y_preds = torch.argmax(teacher_preds[teachable_samples])
                    task_accs[idx] += (y_preds.int() == y[teachable_samples].int()).sum().item()/len(teachable_samples)
                    # print(f'Teachable samples: { (y_preds.int() == y[teachable_samples].int()).sum().item()/len(y[teachable_samples])}')
                if len(non_teachable_samples):
                    tmp_acc = 0
                    for samp in non_teachable_samples:
                        y_preds = torch.argmax(model_preds[samp]) \
                                if model_entropy[samp] <= teacher_entropy[samp] \
                                else torch.argmax(teacher_preds[samp])
                        if y_preds==y[samp].int():
                            tmp_acc += 1
                    tmp_acc /= len(non_teachable_samples)    
                    task_accs[idx] += tmp_acc
                        #task_accs[idx] += (y_preds.int() == y[samp].int()).sum().item()
                        #print(f'NonTeachable samples: {(y_preds.int() == y[non_teachable_samples].int()).sum().item()/len(y[non_teachable_samples])}')
            else:
                y_preds = torch.argmax(model_preds, dim=1)
                task_accs[idx] += (y_preds.int() == y.int()).sum().item()/len(y)
    print(task_accs)
    for i in range(len(dloaders)):
        task_accs[i]  /= len(dloaders[i])
    print(task_accs)            


# def test_with_collab(agents,dloaders,epochs=1,lr=0.001,batch_size=128,device='cpu'):
#     print("post train..")
#     dset = ConcatDataset([dl.dataset for dl in dloaders])
#     dloader = DataLoader(dset,batch_size=batch_size, shuffle=True)
#     task_accs = [0 for _ in range(len(dloaders))]

#     for epoch in tqdm(range(epochs)):
#         for batch in tqdm(dloader):
#             x,y,ds_id = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            
#             model = random.choice(agents)

#             model_preds = model.prediction_means(x)
#             model_entropy = model.predictive_entropy_from_means(model_preds)

#             low_certainty_samples = torch.where(model_entropy>model.entropy_threshold)[0]
#             if low_certainty_samples.numel():
#                 #Get set of which tasks are uncertain
#                 for id_ in set(ds_id[low_certainty_samples]):
#                     teacher = agents[id_]
#                     sample_idxs = torch.where(ds_id==id_)[0]
#                     teacher_preds = teacher.prediction_means(x[sample_idxs])
#                     teacher_entropy = teacher.predictive_entropy_from_means(teacher_preds)
#                     teachable_samples = torch.where(teacher_entropy<teacher.entropy_threshold)[0]
#                     non_teachable_samples = np.delete(np.arange(len(sample_idxs),step=1),teachable_samples.detach().cpu())
                    
#                     #TODO add loss here
#                     y_preds = torch.argmax(teacher_preds[teachable_samples], dim=1)
#                     task_accs[id_] += (y_preds.int() == y[teachable_samples].int()).sum().item()/len(y[teachable_samples])
#                     for smpl in non_teachable_samples:
#                         y_preds = torch.argmax(model_preds[non_teachable_samples], dim=1) \
#                                 if model_entropy[smpl] <= teacher_entropy[smpl] \
#                                 else torch.argmax(teacher_preds[non_teachable_samples], dim=1)
#                         task_accs[id_] += (y_preds.int() == y[non_teachable_samples].int()).sum().item()/len(y[non_teachable_samples])
#             else:
#                 for id_ in set(ds_id):
#                     smpl_idx = torch.where(ds_id==id_)[0]
#                     y_preds = torch.argmax(model_preds[smpl_idx], dim=1)
#                     task_accs[id_] += (y_preds.int() == y[smpl_idx].int()).sum().item()/len(y[smpl_idx])
            
#     for i in range(len(dloaders)):
#         task_accs[i]  /= len(dloaders[i])
#     print(task_accs)




if __name__ == '__main__':
    N_CLASSES = 10
    INPUT_SHAPE = 28*28
    LAYER_WIDTH = 100
    N_HIDDEN_LAYERS = 2
    N_TASKS = 5
    EPOCHS = 5
    BATCH_SIZE = 128
    INITIAL_POSTERIOR_VAR = 1e-8
    #torch.autograd.set_detect_anomaly(True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #create agents
    agents = [MLP(n_tasks=N_TASKS).to(device) \
        for _ in range(N_TASKS)]

    print(list(agents[0].model.named_parameters()))
    sys.exit()
    train_loader, test_loader = get_pmnist_data(n_tasks=N_TASKS,batch_size=BATCH_SIZE)
    print("TO DO: IN POST TRAIN YOU SHOULD ALLOW ANY AGENT TO BE SELECTED, INCLUDING THE AGENT WHOS TASK IS AT HAND")
    for idx,agent in enumerate(agents):
        agent.id = idx
        _train(agent,train_loader[idx],device=device,epochs=EPOCHS)
        _test(agent,test_loader,device=device)
    #test_collab(agents,test_loader,device)

    #temp_test_matchmaking(agents,train_loader,test_loader,epochs=5,device=device)
    #torch.autograd.set_detect_anomaly(True) 
    #sequential_post_train_one_agent(agents,train_loader,test_loader,device=device,epochs=2)
    post_train(agents,train_loader,test_loader,device=device,epochs=2)
    # print("WARNING, YOU STILL NEED TO FIX MM TRAIN/TEST, TASK 0 PERFORMANCE IS 0.0")
    # test_matchmaking(agents,test_loader,device=device)
