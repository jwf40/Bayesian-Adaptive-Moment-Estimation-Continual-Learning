import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from optimizers_lib import fastbgd, bgd# Get requested dataset 
import data
from models.basic_mlp import BasicMLP
from benchmark_methods import *

def _get_data(**kwargs):
    """
    Get the required data loaders for the experiment,
        and the associated additional kwargs (e.g. number of classes in the dataset)
    """
    train_loader, test_loader,dataset_kwargs = getattr(data, f'get_{kwargs["exp"]}')(**kwargs)#get_pmnist_data(n_tasks=N_TASKS,batch_size=BATCH_SIZE)
    kwargs.update(dataset_kwargs)
    if kwargs['graduated']:
        train_loader = [data.GraduatedDataLoader(train_loader)]
    return train_loader, test_loader, kwargs

def mas_main(**kwargs):
    train_loader, test_loader, kwargs = _get_data(**kwargs)
    model = BasicMLP(n_classes=kwargs['n_classes']).to(kwargs["device"])
    method = MAS(model, train_loader, test_loader, **kwargs)
    method.run()

def ewc_main(**kwargs):
    train_loader, test_loader, kwargs = _get_data(**kwargs)
    model = BasicMLP(n_classes=kwargs['n_classes']).to(kwargs["device"])
    method = EWC(model, train_loader, test_loader, **kwargs)
    method.run()

def ewconline_main(**kwargs):
    train_loader, test_loader, kwargs = _get_data(**kwargs)
    model = BasicMLP(n_classes=kwargs['n_classes']).to(kwargs["device"])
    method = EWCOnline(model, train_loader, test_loader, **kwargs)
    method.run()

def synaptic_intelligence_main(**kwargs):
    train_loader, test_loader, kwargs = _get_data(**kwargs)
    model = BasicMLP(n_classes=kwargs['n_classes']).to(kwargs["device"])
    method = SynapticIntelligence(model, train_loader, test_loader, **kwargs)
    method.run()