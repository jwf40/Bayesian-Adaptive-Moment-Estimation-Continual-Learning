import warnings
from fnmatch import fnmatch
from typing import Sequence, Any, Set, List, Tuple, Dict, Union, TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.batchnorm import _NormBase
from torch.utils.data import DataLoader
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.utils import get_layers_and_params, ParamData

from tqdm import tqdm
from .base import BaseCLMethod
from data import GraduatedDataLoader

SynDataType = Dict[str, Dict[str, Union[ParamData, Tensor]]]
ParamDict = Dict[str, ParamData]
EwcDataType = Tuple[ParamDict, ParamDict]


class SynapticIntelligence(BaseCLMethod):
    """Synaptic Intelligence plugin.

    This is the Synaptic Intelligence PyTorch implementation of the
    algorithm described in the paper
    "Continuous Learning in Single-Incremental-Task Scenarios"
    (https://arxiv.org/abs/1806.08568)

    The original implementation has been proposed in the paper
    "Continual Learning Through Synaptic Intelligence"
    (https://arxiv.org/abs/1703.04200).

    This plugin can be attached to existing strategies to achieve a
    regularization effect.

    This plugin will require the strategy `loss` field to be set before the
    `before_backward` callback is invoked. The loss Tensor will be updated to
    achieve the S.I. regularization effect.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: list[Union[DataLoader, GraduatedDataLoader]],
        test_loader: list[DataLoader],        
        eps: float = 0.0000001,
        excluded_parameters: Sequence["str"] = None,
        **kwargs,
    ):
        """Creates an instance of the Synaptic Intelligence plugin.

        :param si_lambda: Synaptic Intelligence lambda term.
            If list, one lambda for each experience. If the list has less
            elements than the number of experiences, last lambda will be
            used for the remaining experiences.
        :param eps: Synaptic Intelligence damping parameter.
        :param device: The device to use to run the S.I. experiences.
            Defaults to "as_strategy", which means that the `device` field of
            the strategy will be used. Using a different device may lead to a
            performance drop due to the required data transfer.
        """

        super().__init__(model, train_loader, test_loader, \
                         file_name = f"SI_ds_{kwargs['exp']}_graduated_{kwargs['graduated']}_lambda_{kwargs['lambda']}",**kwargs)
        
        si_lambda: Union[float, Sequence[float]] = kwargs['lambda']#1.0,
        if excluded_parameters is None:
            excluded_parameters = []
        self.si_lambda = (
            si_lambda if isinstance(si_lambda, (list, tuple)) else [si_lambda]
        )
        self.eps: float = eps
        self.excluded_parameters: Set[str] = set(excluded_parameters)
        self.ewc_data: EwcDataType = (dict(), dict())
        """
        The first dictionary contains the params at loss minimum while the 
        second one contains the parameter importance.
        """

        self.syn_data: SynDataType = {
            "old_theta": dict(),
            "new_theta": dict(),
            "grad": dict(),
            "trajectory": dict(),
            "cum_trajectory": dict(),
        }

    def train(self, loader):
        if self.use_labels:
            self.before_training_exp()
        for ep in tqdm(range(self.epochs)):
            for idx, data in enumerate(tqdm(loader)):
                if not self.use_labels:
                    self.before_training_exp()
                self.before_training_iteration()
                self.optim.zero_grad()
                x, y = data[0].to(self.device), data[1].to(self.device)
                out = self.model(x)
                loss = self.criterion(out, y)
                loss += self.before_backward()
                loss.backward()
                self.optim.step()
                self.after_training_iteration()
                if not self.use_labels and idx %10000==0:
                    # self.params = dict([(n, p.data.clone()) for n,p in self.model.named_parameters()])
                    self.after_training_exp()
                    self.test()

        if self.use_labels:
            # self.params = dict([(n, p.data.clone()) for n,p in self.model.named_parameters()])
            self.after_training_exp()

    def before_training_exp(self, **kwargs):
        SynapticIntelligence.create_syn_data(
            self.model,
            self.ewc_data,
            self.syn_data,
            self.excluded_parameters,
        )

        SynapticIntelligence.init_batch(
            self.model,
            self.ewc_data,
            self.syn_data,
            self.excluded_parameters,
        )

    def before_backward(self, **kwargs):
        exp_id = self.task_counter
        try:
            si_lamb = self.si_lambda[exp_id]
        except IndexError:  # less than one lambda per experience, take last
            si_lamb = self.si_lambda[-1]

        syn_loss = SynapticIntelligence.compute_ewc_loss(
            self.model,
            self.ewc_data,
            self.excluded_parameters,
            lambd=si_lamb,
            device=self.device,
        )

        if syn_loss is not None:
            return syn_loss.to(self.device)
        return 0

    def before_training_iteration(self, **kwargs):
        SynapticIntelligence.pre_update(
            self.model, self.syn_data, self.excluded_parameters
        )

    def after_training_iteration(self, **kwargs):
        SynapticIntelligence.post_update(
            self.model, self.syn_data, self.excluded_parameters
        )

    def after_training_exp(self, **kwargs):
        SynapticIntelligence.update_ewc_data(
            self.model,
            self.ewc_data,
            self.syn_data,
            0.001,
            self.excluded_parameters,
            1,
            eps=self.eps,
        )

    @staticmethod
    @torch.no_grad()
    def create_syn_data(
        model: Module,
        ewc_data: EwcDataType,
        syn_data: SynDataType,
        excluded_parameters: Set[str],
    ):
        params = SynapticIntelligence.allowed_parameters(model, excluded_parameters)

        for param_name, param in params:
            if param_name not in ewc_data[0]:
                # new parameter
                ewc_data[0][param_name] = ParamData(param_name, param.flatten().shape)
                ewc_data[1][param_name] = ParamData(
                    f"imp_{param_name}", param.flatten().shape
                )
                syn_data["old_theta"][param_name] = ParamData(
                    f"old_theta_{param_name}", param.flatten().shape
                )
                syn_data["new_theta"][param_name] = ParamData(
                    f"new_theta_{param_name}", param.flatten().shape
                )
                syn_data["grad"][param_name] = ParamData(
                    f"grad{param_name}", param.flatten().shape
                )
                syn_data["trajectory"][param_name] = ParamData(
                    f"trajectory_{param_name}", param.flatten().shape
                )
                syn_data["cum_trajectory"][param_name] = ParamData(
                    f"cum_trajectory_{param_name}", param.flatten().shape
                )
            elif ewc_data[0][param_name].shape != param.shape:
                # parameter expansion
                ewc_data[0][param_name].expand(param.flatten().shape)
                ewc_data[1][param_name].expand(param.flatten().shape)
                syn_data["old_theta"][param_name].expand(param.flatten().shape)
                syn_data["new_theta"][param_name].expand(param.flatten().shape)
                syn_data["grad"][param_name].expand(param.flatten().shape)
                syn_data["trajectory"][param_name].expand(param.flatten().shape)
                syn_data["cum_trajectory"][param_name].expand(param.flatten().shape)

    @staticmethod
    @torch.no_grad()
    def extract_weights(
        model: Module, target: ParamDict, excluded_parameters: Set[str]
    ):
        params = SynapticIntelligence.allowed_parameters(model, excluded_parameters)

        for name, param in params:
            target[name].data = param.detach().cpu().flatten()

    @staticmethod
    @torch.no_grad()
    def extract_grad(model, target: ParamDict, excluded_parameters: Set[str]):
        params = SynapticIntelligence.allowed_parameters(model, excluded_parameters)

        # Store the gradients into target
        for name, param in params:
            target[name].data = param.grad.detach().cpu().flatten()

    @staticmethod
    @torch.no_grad()
    def init_batch(
        model,
        ewc_data: EwcDataType,
        syn_data: SynDataType,
        excluded_parameters: Set[str],
    ):
        # Keep initial weights
        SynapticIntelligence.extract_weights(model, ewc_data[0], excluded_parameters)
        for param_name, param_trajectory in syn_data["trajectory"].items():
            param_trajectory.data.fill_(0.0)

    @staticmethod
    @torch.no_grad()
    def pre_update(model, syn_data: SynDataType, excluded_parameters: Set[str]):
        SynapticIntelligence.extract_weights(
            model, syn_data["old_theta"], excluded_parameters
        )

    @staticmethod
    @torch.no_grad()
    def post_update(model, syn_data: SynDataType, excluded_parameters: Set[str]):
        SynapticIntelligence.extract_weights(
            model, syn_data["new_theta"], excluded_parameters
        )
        SynapticIntelligence.extract_grad(model, syn_data["grad"], excluded_parameters)

        for param_name in syn_data["trajectory"]:
            syn_data["trajectory"][param_name].data += syn_data["grad"][
                param_name
            ].data * (
                syn_data["new_theta"][param_name].data
                - syn_data["old_theta"][param_name].data
            )

    @staticmethod
    def compute_ewc_loss(
        model,
        ewc_data: EwcDataType,
        excluded_parameters: Set[str],
        device,
        lambd=0.0,
    ):
        params = SynapticIntelligence.allowed_parameters(model, excluded_parameters)

        loss = None
        for name, param in params:
            weights = param.to(device).flatten()  # Flat, not detached
            ewc_data0 = ewc_data[0][name].data.to(device)  # Flat, detached
            ewc_data1 = ewc_data[1][name].data.to(device)  # Flat, detached
            syn_loss: Tensor = torch.dot(ewc_data1, (weights - ewc_data0) ** 2) * (
                lambd / 2
            )

            if loss is None:
                loss = syn_loss
            else:
                loss += syn_loss

        return loss

    @staticmethod
    @torch.no_grad()
    def update_ewc_data(
        net,
        ewc_data: EwcDataType,
        syn_data: SynDataType,
        clip_to: float,
        excluded_parameters: Set[str],
        c=0.0015,
        eps: float = 0.0000001,
    ):
        SynapticIntelligence.extract_weights(
            net, syn_data["new_theta"], excluded_parameters
        )

        for param_name in syn_data["cum_trajectory"]:
            syn_data["cum_trajectory"][param_name].data += (
                c
                * syn_data["trajectory"][param_name].data
                / (
                    np.square(
                        syn_data["new_theta"][param_name].data
                        - ewc_data[0][param_name].data
                    )
                    + eps
                )
            )

        for param_name in syn_data["cum_trajectory"]:
            ewc_data[1][param_name].data = torch.empty_like(
                syn_data["cum_trajectory"][param_name].data
            ).copy_(-syn_data["cum_trajectory"][param_name].data)

        # change sign here because the Ewc regularization
        # in Caffe (theta - thetaold) is inverted w.r.t. syn equation [4]
        # (thetaold - theta)
        for param_name in ewc_data[1]:
            ewc_data[1][param_name].data = torch.clamp(
                ewc_data[1][param_name].data, max=clip_to
            )
            ewc_data[0][param_name].data = syn_data["new_theta"][
                param_name
            ].data.clone()

    @staticmethod
    def explode_excluded_parameters(excluded: Set[str]) -> Set[str]:
        """
        Explodes a list of excluded parameters by adding a generic final ".*"
        wildcard at its end.

        :param excluded: The original set of excluded parameters.

        :return: The set of excluded parameters in which ".*" patterns have been
            added.
        """
        result = set()
        for x in excluded:
            result.add(x)
            if not x.endswith("*"):
                result.add(x + ".*")
        return result

    @staticmethod
    def not_excluded_parameters(
        model: Module, excluded_parameters: Set[str]
    ) -> Sequence[Tuple[str, Tensor]]:
        # Add wildcards ".*" to all excluded parameter names
        result: List[Tuple[str, Tensor]] = []
        excluded_parameters = SynapticIntelligence.explode_excluded_parameters(
            excluded_parameters
        )
        layers_params = get_layers_and_params(model)

        for lp in layers_params:
            if isinstance(lp.layer, _NormBase):
                # Exclude batch norm parameters
                excluded_parameters.add(lp.parameter_name)

        for name, param in model.named_parameters():
            accepted = True
            for exclusion_pattern in excluded_parameters:
                if fnmatch(name, exclusion_pattern):
                    accepted = False
                    break

            if accepted:
                result.append((name, param))

        return result

    @staticmethod
    def allowed_parameters(
        model: Module, excluded_parameters: Set[str]
    ) -> List[Tuple[str, Tensor]]:
        allow_list = SynapticIntelligence.not_excluded_parameters(
            model, excluded_parameters
        )

        result = []
        for name, param in allow_list:
            if param.requires_grad:
                result.append((name, param))

        return result
