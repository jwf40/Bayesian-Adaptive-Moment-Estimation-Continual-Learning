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
        super().__init__(model, train_loader, test_loader,\
                         file_name = f"VCL_ds_{kwargs['exp']}_graduated_{kwargs['graduated']}", **kwargs)
        self.beta = kwargs['vcl_beta']#0.01


    def calculate_accuracy(self, outputs, targets):
        return np.mean(outputs.argmax(dim=-1).cpu().numpy() == targets.cpu().numpy())

    def train(self, loader, T=10, replay=False):
        beta = 0 if replay else self.beta
        
        offset = 0
        output_nodes = 10

        try:
            train_size = len(loader.dataset)
        except AttributeError:
            #Graduated dataloader has no direct dataset
            train_size = loader.size
        elbo = ELBO(self.model, train_size, beta)

        self.model.train()
        for idx, data in enumerate(tqdm(loader)):
            for epoch in range(self.epochs):
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
                loss.backward(retain_graph=True)
                self.optim.step()
                if not self.use_labels and idx %5000==0:
                    self.model.update_prior()
                    self.test()


    def test(self, T=10):
    
        offset = 0
        output_nodes = 10
        
        self.model.train()
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
            self.test_acc_list[idx].append(np.mean(task_accs))



    def run(self, coreset_method=None, update_prior=True):
        num_tasks = len(self.test_loader)

        for task_id in range(len(self.train_loader)):
            trainloader = self.train_loader[task_id]
            self.train(trainloader)

            # Evaluate on old tasks
            self.test()
            if update_prior and self.use_labels:
                self.model.update_prior()
        #print(all_accs)
        self.save(self.test_acc_list, self.root+self.file_name)
        return self.test_acc_list