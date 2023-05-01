import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from .base import BaseCLMethod


class MAS(BaseCLMethod):
    def __init__(self, model, train_loader, test_loader, **kwargs):
        super().__init__(model, train_loader, test_loader,\
                         file_name = f"MAS_ds_{kwargs['exp']}_graduated_{kwargs['graduated']}", **kwargs)
        self.lambda_ = kwargs['mas_lambda']#1.0
        self.alpha_ = kwargs['mas_alpha']#0.5
        self.importances = self.zerolike_params_dict()
        self.saved_params = self.zerolike_params_dict()

    def _update_importances(self, x=None):
        def _get_importances(x=None):
            imps = self.zerolike_params_dict()
            self.model.train()

            iter_struct = self.train_loader[self.task_counter] if x==None else [(x, -1)]
            for data in iter_struct:
                x = data[0].to(self.device)
                self.optim.zero_grad()
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

        self.saved_params = self.copy_params_dict()

        if self.task_counter > 0:
            current_importances = _get_importances(x)
            for n in current_importances.keys():
                curr_shape = current_importances[n].data.shape
                if n not in self.importances:
                    self.importances[n] = current_importances[n].data.clone()
                else:
                    self.importances[n].data = (
                        self.alpha_ * self.importances[n].expand(curr_shape)
                        + (1 - self.alpha_) * current_importances[n].data
                    )
        else:
            self.importances = _get_importances(x)

        self.task_counter += 1
        return

    def _calc_reg(self):
        if self.task_counter == 0:
            return 0.0
        loss_reg = 0.0
        for n, p in self.model.named_parameters():
            if n in self.importances.keys():
                loss_reg += torch.sum(
                    self.importances[n].expand(p.shape)
                    * (p - self.saved_params[n].expand(p.shape)).pow(2)
                )
        return self.lambda_ * loss_reg

    def train(self, loader):        
        xbuffer = []
        ybuffer = []
        for epoch in tqdm(range(self.epochs)):
            rloss = 0
            for idx,batch in enumerate(tqdm(loader)):
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                xbuffer.append(x)
                ybuffer.append(y)
                if len(ybuffer) >= self.buffer_len: 
                    _x = torch.stack(xbuffer).squeeze(1) if self.buffer_len > 1 else x
                    _y = torch.stack(ybuffer).squeeze(1) if self.buffer_len > 1 else y
                    out = self.model(_x)
                    loss = self.criterion(out, _y)
                    loss += self._calc_reg()
                    loss.backward()
                    self.optim.step()
                    xbuffer = []
                    ybuffer = []
                if not self.use_labels and idx %self.test_every==0:
                    # self.params = dict([(n, p.data.clone()) for n,p in self.model.named_parameters()])
                    self._update_importances(x)
                    self.test()

            # print(epoch_loss)
        if self.use_labels:
            # self.params = dict([(n, p.data.clone()) for n,p in self.model.named_parameters()])
            self._update_importances()
