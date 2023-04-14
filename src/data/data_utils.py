import random
from collections import OrderedDict
import torch
import PIL.Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, Normalize
from torchvision.transforms.functional import pil_to_tensor
from models.basic_mlp import BasicMLP
from .pmnist_data import PermutedMNIST

def get_pmnist(batch_size=128, **kwargs):    
    n_tasks= 10 if not kwargs['n_tasks'] else kwargs['n_tasks']
    kwargs = {'n_classes':10, 'hidden':200,\
               'badam_mean_eta': 0.3, 'badam_std': 0.06, \
               'bgd_mean_eta': 1.0, 'bgd_std': 0.06, \
                'mas_lambda': 1.2, 'mas_alpha': 0.4,\
                'ewc_lambda': 1000, 'ewc_decay': 0.9,\
                'si_lambda': 1.0,\
                'tfcl_lambda': 0.4,\
                'vcl_beta': 0.01}
    train_loader = []
    test_loader = []

    transforms = Compose([Normalize((0.1307,), (0.3081,))])

    idx = list(range(28 * 28))
    
    for i in range(n_tasks):
        random.shuffle(idx)
        train_loader.append(DataLoader(PermutedMNIST(train=True, permute_idx=idx, id=i, transform=transforms),\
                                        batch_size=batch_size,num_workers=1, shuffle=True))
        test_loader.append(DataLoader(PermutedMNIST(train=False, permute_idx=idx,  id=i,  transform=transforms),\
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
   

def _displit(dataset, class_split: tuple[list,list], n_splits, flatten=True, normalize=True, **kwargs):
    #Number of tensors in dataset, e.g. X, Y would be 2
    n_tensors = len(next(iter(dataset)))

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
        idx = -1
        for i in range(len(class_split)):
            if y in class_split[i]:
                idx = class_split[i].index(y)
                data[1] = i
                break
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
   
def get_DIsplitmnist(batch_size=128, **kwargs):
    n_tasks= 5 if not kwargs['n_tasks'] else kwargs['n_tasks']
        
    kwargs = {'n_classes':10, 'hidden':200, \
              'badam_mean_eta': 0.2, 'badam_std': 0.01,\
              'bgd_mean_eta': 1, 'bgd_std': 0.01,\
              'ewc_lambda': 1000, 'ewc_decay': 0.9,\
              'mas_lambda': 1.0, 'mas_alpha': 0.5,\
              'si_lambda': 1.0,\
              'tfcl_lambda': 0.5,\
              'vcl_beta': 0.01
                }
    target_1 = [0,2,4,6,8]
    target_2 = [1,3,5,7,9]
    train_dsets = _displit(MNIST(root="~/.torch/data/mnist", train=True, download=True),(target_1, target_2), n_classes=10, n_splits=n_tasks)
    test_dsets = _displit(MNIST(root="~/.torch/data/mnist", train=False, download=True), (target_1, target_2), n_classes=10, n_splits=n_tasks)

    train_loader = [DataLoader(train_dsets[idx], batch_size=batch_size,num_workers=1, shuffle=True) for idx in range(len(test_dsets))]
    test_loader = [DataLoader(test_dsets[idx], batch_size=batch_size,num_workers=1, shuffle=True) for idx in range(len(test_dsets))]

    return (train_loader, test_loader,kwargs)


def get_CIsplitmnist(batch_size=128, **kwargs):
    n_tasks= 5 if not kwargs['n_tasks'] else kwargs['n_tasks']
    kwargs = {'n_classes':10, 'hidden':200, \
              'badam_mean_eta': 0.1, 'badam_std': 0.01,\
              'bgd_mean_eta': 10, 'bgd_std': 0.01,\
              'ewc_lambda': 100, 'ewc_decay': 0.7,\
              'mas_lambda': 1.0, 'mas_alpha': 0.6,\
              'si_lambda': 1.0,\
              'tfcl_lambda': 0.4,\
              'vcl_beta': 0.1
                }
    train_dsets = _split(MNIST(root="~/.torch/data/mnist", train=True, download=True), n_classes=10, n_splits=n_tasks)
    test_dsets = _split(MNIST(root="~/.torch/data/mnist", train=False, download=True), n_classes=10, n_splits=n_tasks)

    train_loader = [DataLoader(train_dsets[idx], batch_size=batch_size,num_workers=1, shuffle=True) for idx in range(n_tasks)]
    test_loader = [DataLoader(test_dsets[idx], batch_size=batch_size,num_workers=1, shuffle=True) for idx in range(n_tasks)]
    
    return (train_loader, test_loader,kwargs)


def get_CIcifar(batch_size=128, **kwargs):
    n_tasks= 5 if not kwargs['n_tasks'] else kwargs['n_tasks']
    kwargs = {'n_classes':10, 'hidden':200, \
              'badam_mean_eta': 0.1, 'badam_std': 0.01,\
              'bgd_mean_eta': 10, 'bgd_std': 0.01,\
              'ewc_lambda': 100, 'ewc_decay': 0.7,\
              'mas_lambda': 1.0, 'mas_alpha': 0.6,\
              'si_lambda': 1.0,\
              'tfcl_lambda': 0.4,\
              'vcl_beta': 0.1
                }
    train_dsets = _split(CIFAR10(root="~/.torch/data/cifar10", train=True, download=True), n_classes=10, n_splits=n_tasks, flatten=False)
    test_dsets = _split(CIFAR10(root="~/.torch/data/cifar10", train=False, download=True), n_classes=10, n_splits=n_tasks, flatten=False)

    train_loader = [DataLoader(train_dsets[idx], batch_size=batch_size,num_workers=1, shuffle=True) for idx in range(n_tasks)]
    test_loader = [DataLoader(test_dsets[idx], batch_size=batch_size,num_workers=1, shuffle=True) for idx in range(n_tasks)]
    
    return (train_loader, test_loader,kwargs)


def get_DIcifar(batch_size=128, **kwargs):
    n_tasks= 5 if not kwargs['n_tasks'] else kwargs['n_tasks']
        
    kwargs = {'n_classes':10, 'hidden':200, \
              'badam_mean_eta': 0.2, 'badam_std': 0.01,\
              'bgd_mean_eta': 1, 'bgd_std': 0.01,\
              'ewc_lambda': 1000, 'ewc_decay': 0.9,\
              'mas_lambda': 1.0, 'mas_alpha': 0.5,\
              'si_lambda': 1.0,\
              'tfcl_lambda': 0.5,\
              'vcl_beta': 0.01
                }
    target_1 = [0,2,4,6,8]
    target_2 = [1,3,5,7,9]
    train_dsets = _displit(CIFAR10(root="~/.torch/data/cifar10", train=True, download=True),(target_1, target_2), n_classes=10, n_splits=n_tasks, flatten=False)
    test_dsets = _displit(CIFAR10(root="~/.torch/data/cifar10", train=False, download=True), (target_1, target_2), n_classes=10, n_splits=n_tasks, flatten=False)

    train_loader = [DataLoader(train_dsets[idx], batch_size=batch_size,num_workers=1, shuffle=True) for idx in range(len(test_dsets))]
    test_loader = [DataLoader(test_dsets[idx], batch_size=batch_size,num_workers=1, shuffle=True) for idx in range(len(test_dsets))]

    return (train_loader, test_loader,kwargs)