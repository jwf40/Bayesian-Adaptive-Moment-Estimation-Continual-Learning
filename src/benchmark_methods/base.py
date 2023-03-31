import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
class BaseCLMethod:
    def __init__(self, model, train_loader, test_loader, **kwargs):
        self.model = model.to(kwargs['device'])
        self.train_loader = train_loader
        self.test_loader =test_loader
        self.epochs = kwargs['epochs']
        self.device = kwargs['device']       
        self.use_labels = kwargs['labels']
        self.optim = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)#, betas=(0.9,0.999)
        self.criterion = nn.CrossEntropyLoss()
        self.loss_per_iter = []
        self.test_acc_per_iter = []
        self.task_counter = 0

    def run(self):            
        for task in tqdm(self.train_loader):
            self.train(task)
            self.test()

    def train(self, loader):
        raise NotImplementedError

    def test(self):        
        for idx,task in enumerate(self.test_loader):
            task_acc = 0
            for data in task:
                x, y = data[0].to(self.device), data[1].to(self.device)
                preds = self.model(x)
                task_acc += ((torch.argmax(preds, dim=1)==y).sum())/len(y)
            task_acc /= len(task)
            print(f'Task {idx} Accuracy: {task_acc}')
                

    def save(self):
        raise NotImplementedError

    # def zerolike_params_dict(self):
    #     return dict([(n, torch.zeros_like(p))
    #             for n, p in self.model.named_parameters()])
    
    def zerolike_params_dict(self, model=None):
        if model==None:
            model = self.model
        return dict([(n, torch.zeros_like(p))
                for n, p in model.named_parameters()])

    def copy_params_dict(self):
        return dict([(n, p.data.clone())
                for n, p in self.model.named_parameters()])