from collections import defaultdict
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from .base import BaseCLMethod

class EWC(BaseCLMethod):
    def __init__(self, model, train_loader, test_loader, **kwargs):
        super().__init__(model, train_loader, test_loader, **kwargs)
        self.lambda_ = 1000
        self.importances = defaultdict(dict)
        self.saved_params = defaultdict(dict)

    def _calc_reg(self):
        if self.task_counter == 0:
            print("Returning becuase its task:", self.task_counter)
            return 0.0
        
        penalty = torch.tensor(0).float().to(self.device)

        for experience in range(self.task_counter):
            for n,p in self.model.named_parameters():
                if n not in self.saved_params[experience]:
                    continue
                saved_param = self.saved_params[experience][n]
                imp = self.importances[experience][n]
                shape =  p.shape
                penalty += (imp.expand(shape) * \
                    (p -saved_param.expand(shape)).pow(2)).sum()
        return self.lambda_ * penalty

    def _compute_importances(self, data=None):
        _importances = self.zerolike_params_dict()
        iter_struct = self.train_loader[self.task_counter] if not data else [data]
        for data in iter_struct:
            x, y = data[0].to(self.device), data[1].to(self.device)
            self.optim.zero_grad()
            out = self.model(x)
            loss = self.criterion(out,y)
            loss.backward()

            for (n1, p), (n2, imp) in zip(
                self.model.named_parameters(), _importances.items()
            ):
                assert n1 == n2
                if p.grad is not None:
                    imp.data += p.grad.data.clone().pow(2)

                    # average over mini batch length
        for _, imp in _importances.items():
            imp.data /= float(len(iter_struct))
        return _importances
    
    def _update_importances(self, data=None):
        self.importances[self.task_counter] = self._compute_importances(data)
        self.saved_params[self.task_counter] = self.copy_params_dict()
        # if self.task_counter>0:
        #     del self.saved_params[self.task_counter-1]
        self.task_counter+=1
        return

    def train(self, loader):
        for ep in tqdm(range(self.epochs)):
            epoch_loss = 0.0
            for idx, data in enumerate(loader):
                self.optim.zero_grad()
                x, y = data[0].to(self.device), data[1].to(self.device)
                out = self.model(x)
                loss = self.criterion(out, y)
                epoch_loss += loss
                loss += self._calc_reg()
                loss.backward()
                self.optim.step()

                if not self.use_labels:
                    #self.params = dict([(n, p.data.clone()) for n,p in self.model.named_parameters()])
                    self._update_importances((x,y))

            # print(epoch_loss)
        if self.use_labels:
            #self.params = dict([(n, p.data.clone()) for n,p in self.model.named_parameters()])
            self._update_importances()