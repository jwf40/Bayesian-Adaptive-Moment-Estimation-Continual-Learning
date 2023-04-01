import numpy as np
from tqdm import tqdm
import torch
import torchvision
from torch import nn
from torch.optim import Adam, SGD
import torch.nn.functional as F

from .base import BaseCLMethod
from .coresets import *

class ELBO(nn.Module):

    def __init__(self, model, train_size, beta):
        super().__init__()
        self.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.beta = beta
        self.train_size = train_size

    def forward(self, outputs, targets, kl):
        assert not targets.requires_grad
        # print(F.nll_loss(outputs, targets, reduction='mean'), self.beta * kl / self.num_params)
        return F.nll_loss(outputs, targets, reduction='mean') + self.beta * kl / self.num_params


class VCL(BaseCLMethod):
    def __init__(self, model, train_loader, test_loader, **kwargs):
        super().__init__(model, train_loader, test_loader, **kwargs)
        self.beta = 0.01

    def calculate_accuracy(self, outputs, targets):
        return np.mean(outputs.argmax(dim=-1).cpu().numpy() == targets.cpu().numpy())

    def train(self, loader, T=10, replay=False):
        beta = 0 if replay else self.beta
        
        offset = 0
        output_nodes = 10


        train_size = len(loader.dataset)
        elbo = ELBO(self.model, train_size, beta)

        self.model.train()
        for epoch in tqdm(range(self.epochs)):
            for data in loader:
                self.optim.zero_grad()
                inputs, targets = data[0].to(self.device), data[1].to(self.device)
                targets -= offset
                outputs = torch.zeros(inputs.shape[0], output_nodes, T, device=self.device)

                for i in range(T):
                    net_out = self.model(inputs)
                    outputs[:, :, i] = F.log_softmax(net_out, dim=-1)

                log_output = torch.logsumexp(outputs, dim=-1) - np.log(T)
                kl = self.model.get_kl()
                loss = elbo(log_output, targets, kl)
                loss.backward()
                self.optim.step()
                if not self.use_labels:
                    self.model.update_prior()


    def test(self, T=10):
    
        offset = 0
        output_nodes = 10
        
        self.model.train()
        accs = []
        for idx, task in enumerate(self.test_loader):
            task_accs = []
            for data in task:
                inputs, targets = data[0].to(self.device), data[1].to(self.device)
                targets -= offset
                outputs = torch.zeros(inputs.shape[0], output_nodes, T, device=self.device)

                for i in range(T):
                    with torch.no_grad():
                        net_out = self.model(inputs)
                    outputs[:, :, i] = F.log_softmax(net_out, dim=-1)

                log_output = torch.logsumexp(outputs, dim=-1) - np.log(T)
                task_accs.append(self.calculate_accuracy(log_output, targets))
            accs.append(np.mean(task_accs))
        return accs


    def run(self, coreset_method=None, update_prior=True):
        num_tasks = len(self.test_loader)
        coreset_list = []
        all_accs = np.empty(shape=(num_tasks, num_tasks))
        all_accs.fill(np.nan)
        for task_id in range(num_tasks):
            trainloader = self.train_loader[task_id]
            print("Starting Task", task_id + 1)
            offset = 0            
            self.train(trainloader)
            print("Done Training Task", task_id + 1)

            # Attach a new coreset
            if coreset_method:
                coreset_method(coreset_list, trainloader, num_samples=200)

                # Replay old tasks using coresets
                for task in range(task_id + 1):
                    print("Replaying Task", task + 1)
                    self.corset_train(coreset_list[task], replay=True)
            # Evaluate on old tasks
            all_accs[task_id] = self.test()
            print(all_accs[task_id])
            if update_prior and self.use_labels:
                self.model.update_prior()
        #print(all_accs)
        return all_accs