from collections import defaultdict, namedtuple
from typing import NamedTuple
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from .base import BaseCLMethod
from fnmatch import fnmatch

class LayerAndParameter(NamedTuple):
    layer_name: str
    layer: torch.nn.Module
    parameter_name: str
    parameter: torch.Tensor

class SI(BaseCLMethod):
    def __init__(self, model, train_loader, test_loader, **kwargs):
        super().__init__(model, train_loader, test_loader, **kwargs)
        self.lambda_ = 1.0
        self.eps = 0.0000001
        self.importances = defaultdict(dict)
        self.saved_params = defaultdict(dict)
        self.excluded_params = set([])
        self.ewc_data = (dict(), dict())
        
        self.syn_data = {
            "old_theta": dict(),
            "new_theta": dict(),
            "grad": dict(),
            "trajectory": dict(),
            "cum_trajectory": dict(),
        }
    
    @torch.no_grad()
    def _init_experience(self):        
        params = self._get_allowed_params()
        for param_name,param in params:
            if param_name not in self.ewc_data[0]:
                self.ewc_data[0][param_name] = torch.zeros_like(param.flatten())
                self.ewc_data[1][param_name] = torch.zeros_like(param.flatten())
                self.syn_data["old_theta"][param_name] = torch.zeros_like(param.flatten())
                self.syn_data["new_theta"][param_name] = torch.zeros_like(param.flatten())
                self.syn_data["grad"][param_name] = torch.zeros_like(param.flatten())
                self.syn_data["trajectory"][param_name] = torch.zeros_like(param.flatten())
                self.syn_data["cum_trajectory"][param_name] = torch.zeros_like(param.flatten())
            elif self.ewc_data[0][param_name].shape != param.shape:
                self.ewc_data[0][param_name].expand(param.flatten().shape)
                self.ewc_data[1][param_name].expand(param.flatten().shape)
                self.syn_data["old_theta"][param_name].expand(param.flatten().shape)
                self.syn_data["new_theta"][param_name].expand(param.flatten().shape)
                self.syn_data["grad"][param_name].expand(param.flatten().shape)
                self.syn_data["trajectory"][param_name]\
                    .expand(param.flatten().shape)
                self.syn_data["cum_trajectory"][param_name]\
                    .expand(param.flatten().shape)
        
        self._extract_weights(self.ewc_data[0])
        for n,p_traj in self.syn_data["trajectory"].items():
            p_traj.data.fill_(0.0)

    def _extract_grad(self, target):
        params = self._get_allowed_params()
        for n,p in params:
            target[n].data = p.grad.detach().cpu().flatten()

    def _extract_weights(self, target):
        params = self._get_allowed_params()
        try:
            for n, p in params:
                target[n].data = p.grad.detach().cpu().flatten()
        except AttributeError:
            pass


    def _get_allowed_params(self):
        def _get_excluded_params():
            result = set()
            for x in self.excluded_params:
                result.add(x)
                if not x.enswith("*"):
                    result.add(x+".*")
            return result

        def _get_layers_and_params(module=self.model, prefix=""):              
            result = []
            for n,p in module.named_parameters(recurse=False):
                result.append(LayerAndParameter(prefix[:-1], module, prefix+n,p))
            for ln, layer in module.named_modules():
                if layer==module:
                    continue
                layer_complete_name = prefix+ln+"."

                result+=_get_layers_and_params(module=layer,prefix=layer_complete_name)
            
            return result
        
        params = []
        allowed_list = []
        excluded_params = _get_excluded_params()
        layer_params = _get_layers_and_params()
        for lp in layer_params:
            if isinstance(lp.layer, torch.nn.modules.batchnorm._NormBase):
                excluded_params.add(lp.parameter_name)
        for name, param in self.model.named_parameters():
            accepted=True
            for pattern in excluded_params:
                if fnmatch(name,pattern):
                    accepted=False
                    break
            if accepted:
                allowed_list.append((name, param))
        
        for name, param in allowed_list:
            if param.requires_grad:
                params.append((name, param))
        return params

    @torch.no_grad()
    def _before_iteration(self):
        self._extract_weights(self.syn_data["old_theta"])

    def _after_iteration(self):
        self._extract_weights(self.syn_data["new_theta"])
        self._extract_grad(self.syn_data["grad"])
        for n in self.syn_data["trajectory"]:
            self.syn_data["trajectory"][n].data += self.syn_data["grad"][n].data \
                    * (self.syn_data["new_theta"][n].data-self.syn_data["old_theta"][n].data)


    def _calc_reg(self):
        lambda_ = self.lambda_
        params = self._get_allowed_params()
        loss = 0.0
        for n,p in params:
            weights = p.to(self.device).flatten()
            ewc_data0 = self.ewc_data[0][n].data.to(self.device)
            ewc_data1 = self.ewc_data[1][n].data.to(self.device)
            syn_loss = torch.dot(ewc_data1, (weights-ewc_data0)**2)\
                                * (lambda_/2)
            loss+=syn_loss
        return loss    
        
   
    def _update_importances(self, c=0.0015,data=None):
        self._extract_weights(self.syn_data['new_theta'])
        for n in self.syn_data["cum_trajectory"]:
            self.syn_data["cum_trajectory"][n].data +=(
                c
                * self.syn_data["trajectory"][n].data
                / (
                    np.square(
                        self.syn_data["new_theta"][n].data
                        - self.ewc_data[0][n].data
                    )
                    + self.eps
                )
            )
        for param_name in self.syn_data["cum_trajectory"]:
            self.ewc_data[1][param_name].data = torch.empty_like(
                self.syn_data["cum_trajectory"][param_name].data
            ).copy_(-self.syn_data["cum_trajectory"][param_name].data)

        # change sign here because the Ewc regularization
        # in Caffe (theta - thetaold) is inverted w.r.t. syn equation [4]
        # (thetaold - theta)
        for param_name in self.ewc_data[1]:
            self.ewc_data[1][param_name].data = torch.clamp(
                self.ewc_data[1][param_name].data, max=0.001
            )
            self.ewc_data[0][param_name].data = \
                self.syn_data["new_theta"][param_name].data.clone()

        self.task_counter+=1
        return

    def train(self, loader):
        if self.use_labels:
            self._init_experience()
        for ep in tqdm(range(self.epochs)):
            epoch_loss = 0.0
            for idx, data in enumerate(loader):
                if not self.use_labels:
                    self._init_experience()
                self._before_iteration()
                self.optim.zero_grad()
                x, y = data[0].to(self.device), data[1].to(self.device)
                out = self.model(x)
                loss = self.criterion(out, y)
                epoch_loss += loss
                loss += self._calc_reg()
                loss.backward()
                self.optim.step()
                self._after_iteration()
                if not self.use_labels:
                    #self.params = dict([(n, p.data.clone()) for n,p in self.model.named_parameters()])
                    self._update_importances()

            # print(epoch_loss)
        if self.use_labels:
            #self.params = dict([(n, p.data.clone()) for n,p in self.model.named_parameters()])
            self._update_importances()
