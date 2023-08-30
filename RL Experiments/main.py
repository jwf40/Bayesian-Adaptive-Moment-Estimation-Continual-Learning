import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
import numpy as np
import random

from avalanche.benchmarks.classic import PermutedMNIST, SplitMNIST, SplitFMNIST,SplitCIFAR10
from avalanche.models import SimpleMLP, SimpleCNN, MTSimpleCNN
from avalanche.training import Naive, MAS, EWC, SynapticIntelligence
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics

from avalanche_rl.benchmarks.rl_benchmark_generators import gym_benchmark_generator
from avalanche_rl.models.actor_critic import ActorCriticMLP
from avalanche_rl.training.strategies import A2CStrategy

from utils import get_exp_conf
from avalanche.logging import InteractiveLogger, CSVLogger
from optimizers_lib import badam, bgd
from badam import BAdam
from vcl import VCL

import pandas as pd

import optuna
import wandb
import sys
"""
Select which experiment or method you would like
"""
exps = ['cartpole']#'',,,'
methods = ['Naive', 'BAdam','BGD','VCL','BAdam','BGD','EWC', 'MAS','SI']


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
"""
Good Method Parameters
"""
badam_etas = {'cartpole': 0.01}
bgd_etas = {'cartpole': 1.0}
badam_stds = {'cartpole': 0.011}
bgd_stds = {'cartpole': 0.01}
mas_lambdas = {'cartpole': 1.0}
vcl_epochs = {'cartpole': 100}



for exp in exps:    
    for run in range(1, 2):
        for idx,method in enumerate(methods):       
            wandb.init(project=f"{exp}", name=f'{method}_{run}')
            random.seed(run)
            np.random.seed(run)
            torch.manual_seed(run)
            csv_logger = CSVLogger(log_folder=f'results/{method}_{exp}_run_{run}')
            interactive_logger = InteractiveLogger()
            kwargs=get_exp_conf(exp)
            eval_plugin = EvaluationPlugin(
                accuracy_metrics(
                    minibatch=False, epoch=True, experience=True, stream=True
                ),
                forgetting_metrics(experience=True),
                loggers=[interactive_logger, csv_logger],
            )

            model = ActorCriticMLP(num_inputs=4, num_actions=2, actor_hidden_sizes=1024, critic_hidden_sizes=1024)
            # CRL Benchmark Creation
            scenario = gym_benchmark_generator(['CartPole-v1'], n_experiences=1, n_parallel_envs=1, 
                eval_envs=['CartPole-v1'])
            
            # Prepare for training & testing
            optimizer = SGD(model.parameters(), lr=0.01)#
            if method == 'BAdam':
                optimizer = badam(model, mean_eta=badam_etas[exp], std_init=badam_stds[exp])
            elif method=='BGD':
                optimizer = bgd(model, mean_eta=bgd_etas[exp], std_init=bgd_stds[exp])
            elif method=='VCL':
                optimizer = Adam(model.parameters(), lr=0.001)#

            criterion = CrossEntropyLoss()
            
            strategy = A2CStrategy(model, optimizer, per_experience_steps=10000, max_steps_per_rollout=5, 
            device=device, eval_every=1000, eval_episodes=10)            

            method_objects = {
                    'BAdam': BAdam(model, optimizer, criterion,\
                    train_mb_size=kwargs['batch_size'], train_epochs=20, eval_mb_size=kwargs['batch_size'], device=device, evaluator=eval_plugin),
                    'BGD': BAdam(kwargs['model'], optimizer, criterion,\
                    train_mb_size=kwargs['batch_size'], train_epochs=kwargs['epochs'], eval_mb_size=kwargs['batch_size'], device=device, evaluator=eval_plugin),                    
                    'EWC': EWC(kwargs['model'],optimizer, criterion,ewc_lambda=10, mode='online',decay_factor=0.9,train_mb_size=kwargs['batch_size'], \
                        train_epochs=kwargs['epochs'], eval_mb_size=kwargs['batch_size'], device=device, evaluator=eval_plugin),
                    'Naive': Naive(kwargs['model'], optimizer, criterion, \
                            train_mb_size=kwargs['batch_size'], train_epochs=kwargs['epochs'], eval_mb_size=kwargs['batch_size'], device=device, evaluator=eval_plugin),
                    'MAS': MAS(kwargs['model'], optimizer, criterion, lambda_reg=0.1,train_mb_size=kwargs['batch_size'], \
                        train_epochs=kwargs['epochs'], eval_mb_size=kwargs['batch_size'], device=device, evaluator=eval_plugin),                
                    'SI': SynapticIntelligence(kwargs['model'],optimizer, criterion,si_lambda=1.0, train_mb_size=kwargs['batch_size'], \
                        train_epochs=kwargs['epochs'], eval_mb_size=kwargs['batch_size'], device=device, evaluator=eval_plugin),
                    'VCL': VCL(kwargs['vcl_model'],optimizer, criterion,train_mb_size=kwargs['batch_size'], \
                        train_epochs=vcl_epochs[exp], eval_mb_size=kwargs['batch_size'], device=device, evaluator=eval_plugin)
            }
            
            cl_strategy = strategy#method_objects[method]
            # train and test loop over the stream of experiences
            results = []
            for train_exp in scenario.train_stream:
                mean_stds = cl_strategy.train(train_exp)
                results.append(cl_strategy.eval(scenario.eval_stream)['Top1_Acc_Stream/eval_phase/test_stream/Task000'])
            if isinstance(mean_stds, pd.DataFrame):
                mean_stds.to_csv(f"{method}_{run}_param_changes.csv", index=False)            
                mean, std = wandb.Table(dataframe=pd.DataFrame(mean_stds['mean'])), wandb.Table(dataframe=pd.DataFrame(mean_stds['std']))

            else:
                mean, std = None, None

            wandb.log(
                {
                    'Test Accuracy': results,
                    'Mean Per Epoch': mean, 
                    'Std Per Epoch': std
                }
            )
            wandb.finish()
            sys.exit()
