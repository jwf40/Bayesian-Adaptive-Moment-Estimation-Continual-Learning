from typing import Optional, Sequence, List, Union, Iterable
import numpy as np
import torch
from copy import deepcopy
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer, SGD
import torch.nn.functional as F
from avalanche.models.pnn import PNN
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.plugins import (
    SupervisedPlugin,
    CWRStarPlugin,
    ReplayPlugin,
    GenerativeReplayPlugin,
    TrainGeneratorAfterExpPlugin,
    GDumbPlugin,
    LwFPlugin,
    AGEMPlugin,
    GEMPlugin,
    EWCPlugin,
    EvaluationPlugin,
    SynapticIntelligencePlugin,
    CoPEPlugin,
    GSS_greedyPlugin,
    LFLPlugin,
    MASPlugin,
    BiCPlugin,
    MIRPlugin,
)
from avalanche.training.templates.base import BaseTemplate
from avalanche.training.templates import SupervisedTemplate
from avalanche.models.generator import MlpVAE, VAE_loss
from avalanche.logging import InteractiveLogger
import pandas as pd

class ELBO(Module):

    def __init__(self, model, beta):
        super().__init__()
        self.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.beta = beta

    def forward(self, outputs, targets, kl):
        assert not targets.requires_grad
        # print(F.nll_loss(outputs, targets, reduction='mean'), self.beta * kl / self.num_params)
        return F.nll_loss(outputs, targets, reduction='mean') + self.beta * kl / self.num_params


class VCL(SupervisedTemplate):
    def __init__(self,
        model: Module,
        optimizer: Optimizer,
        criterion=CrossEntropyLoss(),
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = None,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = default_evaluator(),
        eval_every=-1,
        **base_kwargs):

        super().__init__(model,
            optimizer,
            criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs)
        
        self.beta = 1
        self.elbo = ELBO(self.model, self.beta)
        for n,p in self.model.named_parameters():
            print(n)
        print("done")

    def calculate_accuracy(self, outputs, targets):
        return np.mean(outputs.argmax(dim=-1).cpu().numpy() == targets.cpu().numpy())


    def _after_training_exp(self, **kwargs):
        self.model.update_prior()

    def training_epoch(self, **kwargs):
        T = 10
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break
            self.optimizer.zero_grad()
            self._unpack_minibatch()
            self.model._to(self.device)     
            self._before_training_iteration(**kwargs)
            outputs = torch.zeros(self.mb_x.shape[0], 10, T, device=self.device)
            self.loss=0.0
            self._before_forward(**kwargs)
            for i in range(T):
                

                net_out = self.model(self.mb_x)
                outputs[:, :, i] = F.log_softmax(net_out, dim=-1)
            self._after_forward(**kwargs)
            log_output = torch.logsumexp(outputs, dim=-1) - np.log(T)
            self.mb_output = log_output
            kl = self.model.get_kl()
            self.loss = self.elbo(log_output, self.mb_y, kl)
            self._before_backward(**kwargs)
            self.loss.backward(retain_graph=True)
            self._after_backward(**kwargs)
            self._before_update(**kwargs)
            self.optimizer.step()
                        
            self._after_training_iteration(**kwargs)