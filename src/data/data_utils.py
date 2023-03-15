import random
from collections import OrderedDict
import torch
import PIL.Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torchvision.transforms.functional import pil_to_tensor
from models.basic_mlp import BasicMLP
from .pmnist_data import PermutedMNIST

def get_pmnist(n_tasks, batch_size=128, **kwargs):    
    kwargs = {'n_classes':10}
    train_loader = []
    test_loader = []

    idx = list(range(28 * 28))
    for i in range(n_tasks):
        random.shuffle(idx)
        train_loader.append(DataLoader(PermutedMNIST(train=True, permute_idx=idx, id=i),\
                                        batch_size=batch_size,num_workers=1, shuffle=True))
        test_loader.append(DataLoader(PermutedMNIST(train=False, permute_idx=idx,  id=i),\
                                      batch_size=batch_size))
        
    return (train_loader, test_loader,kwargs)

def _split(dataset ,n_classes,n_splits, flatten=True, normalize=True, **kwargs):
    assert n_classes % n_splits == 0, 'Error, not an even split!'
    #Number of tensors in dataset, e.g. X, Y would be 2
    n_tensors = len(next(iter(dataset)))
    #number of classes in each task
    class_cnt = n_classes // n_splits

    #Dictionary of each tensor split into the correct subclasses, one for each tensor
    tensor_dict = {idx: [[] for _ in range(n_splits)] \
                   for idx in range(n_tensors)}
    
    for data in dataset:
        #Infer class
        data = list(data)
        data[0] = pil_to_tensor(data[0])
        data[0] = data[0].to(torch.float32)

        if normalize:
            data[0] /= 255
        if flatten:
            data[0] = data[0].view(-1)
        y = data[1]
        idx = y//class_cnt
        #Populate tensor dict
        for tensor_idx in tensor_dict.keys():
            tensor_dict[tensor_idx][idx].append(data[tensor_idx])
    
    return_li = []
    for idx in range(n_splits):
        sub_li = []
        for key in sorted(tensor_dict.keys()):
            #Get list of all Tensors for a given split.
            try:
                dtype = torch.float32 if key != 1 else torch.int64
                sub_li.append(torch.tensor(tensor_dict[key][idx], dtype=dtype))
                
            except:
                sub_li.append(torch.stack(tensor_dict[key][idx]))
        #Convert list to tensor dataset, unroll list of tensors as args
        return_li.append(TensorDataset(*sub_li))    
    return return_li
   
        
   
def get_DIsplitmnist(n_tasks, batch_size=128, **kwargs):
    kwargs = {'n_classes':2}
    train_dsets = _split(MNIST(root="~/.torch/data/mnist", train=True, download=True), n_classes=10, n_splits=n_tasks)
    test_dsets = _split(MNIST(root="~/.torch/data/mnist", train=False, download=True), n_classes=10, n_splits=n_tasks)

    train_loader = [DataLoader(train_dsets[idx], batch_size=batch_size,num_workers=1, shuffle=True) for idx in range(n_tasks)]
    test_loader = [DataLoader(test_dsets[idx], batch_size=batch_size,num_workers=1, shuffle=True) for idx in range(n_tasks)]
    
    return (train_loader, test_loader,kwargs)


def get_CIsplitmnist(n_tasks, batch_size=128, **kwargs):
    kwargs = {'n_classes':10}
    train_dsets = _split(MNIST(root="~/.torch/data/mnist", train=True, download=True), n_classes=10, n_splits=n_tasks)
    test_dsets = _split(MNIST(root="~/.torch/data/mnist", train=False, download=True), n_classes=10, n_splits=n_tasks)

    train_loader = [DataLoader(train_dsets[idx], batch_size=batch_size,num_workers=1, shuffle=True) for idx in range(n_tasks)]
    test_loader = [DataLoader(test_dsets[idx], batch_size=batch_size,num_workers=1, shuffle=True) for idx in range(n_tasks)]
    
    return (train_loader, test_loader,kwargs)
