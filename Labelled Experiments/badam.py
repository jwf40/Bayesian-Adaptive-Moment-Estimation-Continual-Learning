from typing import Optional, Sequence, List, Union
import numpy as np
import torch
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer, SGD

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

class BAdam(SupervisedTemplate):
    """Naive finetuning.

    The simplest (and least effective) Continual Learning strategy. Naive just
    incrementally fine tunes a single model without employing any method
    to contrast the catastrophic forgetting of previous knowledge.
    This strategy does not use task identities.

    Naive is easy to set up and its results are commonly used to show the worst
    performing baseline.
    """

    def __init__(
        self,
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
        **base_kwargs
    ):
        """
        Creates an instance of the Naive strategy.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param base_kwargs: any additional
            :class:`~avalanche.training.BaseTemplate` constructor arguments.
        """
        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )
        self.running_params = pd.DataFrame(columns=['Mean', 'Std'])

    def training_epoch(self, **kwargs):
        """Training epoch.

        :param kwargs:
        :return:
        """
        mean, std = [], []
        for group in self.optimizer.param_groups:
            mean.append(torch.mean(group["mean_param"]).detach().cpu())
            std.append(torch.mean(group["std_param"]).detach().cpu())
        _opt = str(type(self.optimizer)).replace('>', '').replace("'","").split(".")[-1]
        self.running_params._append({'Mean':np.mean(mean),'Std': np.mean(std)}, ignore_index=True)
        self.running_params.to_csv(f"{_opt}_param_changes.csv", index=False)            

        for self.mbatch in self.dataloader:
            if self._stop_training:
                break
            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)
            self.optimizer.zero_grad()
            self.loss = 0
            for mc_iter in range(self.optimizer.mc_iters):
                self.optimizer.randomize_weights()         
                # Forward
                self._before_forward(**kwargs)
                self.mb_output = self.forward()
                self._after_forward(**kwargs)
                # Loss & Backward
                self.loss += self.criterion()
                self.optimizer.zero_grad()
                self._before_backward(**kwargs)
                self.backward(retain_graph=True)
                self._after_backward(**kwargs)
                self.optimizer.aggregate_grads(len(self.mbatch))
            # Optimization step
            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)
