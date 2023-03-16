import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from .base import BaseCLMethod

class MAS(BaseCLMethod):
    def __init__(self, model, train_loader, test_loader, **kwargs):
        super().__init__(model, train_loader, test_loader, **kwargs)
        self.lambda_ = 1.0
        self.alpha_ = 0.5
        self.importances = self.zerolike_params_dict()
        self.last_params = self.zerolike_params_dict()

    def run(self):            
        for task in self.train_loader:
            self.train(task)
            self.test()

    def _update_importances(self, x=None):
        def _get_importances(x=None):
            imps = self.zerolike_params_dict()
            self.model.train()
            
            iter_struct = self.train_loader[self.task_counter] if not x else [(x, -1)]
            for data in iter_struct:
                x = data[0].to(self.device)
                out = self.model(x)
                loss = torch.norm(out, p="fro", dim=1).pow(2).mean()
                loss.backward()
                for n, p in self.model.named_parameters():
                    if p.requires_grad:
                        # In multi-head architectures, the gradient is going
                        # to be None for all the heads different from the
                        # current one.
                        if p.grad is not None:
                            imps[n].data += p.grad.abs()

            for k in imps.keys():
                imps[k].data /= float(len(iter_struct))
            return imps

        if self.task_counter > 0:
            current_importances = _get_importances(x)
            for n in current_importances.keys():
                curr_shape = current_importances[n].data.shape
                if n not in self.importances:
                    self.importances[n] = current_importances[n].data.clone()
                else:
                    self.importances[n].data = (
                        self.alpha_ * self.importances[n].expand(curr_shape)
                        + (1-self.alpha_)*current_importances[n].data
                    )
        else:
            self.importances = _get_importances(x)
        
        self.task_counter += 1
        return


    def _calc_reg(self):
        if self.task_counter == 0:
            return 0
        loss_reg = 0.0
        for n,p in self.model.named_parameters():
            if n in self.importances.keys():
                loss_reg += torch.sum(
                    self.importances[n].expand(p.shape) *
                    (p - self.last_params[n].expand(p.shape)).pow(2)
                )
        return self.lambda_*loss_reg
    
    def train(self, loader):
        for ep in range(self.epochs):
            epoch_loss = 0.0
            for idx, data in enumerate(loader):
                self.optim.zero_grad()
                x, y = data[0].to(self.device), data[1].to(self.device)
                out = self.model(x)
                loss = self.criterion(out, y)
                epoch_loss += loss
                #loss += self._calc_reg()
                loss.backward()
                self.optim.step()

                if not self.use_labels:
                    self.params = dict([(n, p.data.clone()) for n,p in self.model.named_parameters()])
                    self._update_importances(x)

            print(epoch_loss)
        if self.use_labels:
            self.params = dict([(n, p.data.clone()) for n,p in self.model.named_parameters()])
            self._update_importances()

                
                

