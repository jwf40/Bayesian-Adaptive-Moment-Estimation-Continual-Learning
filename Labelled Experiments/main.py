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

from utils import get_exp_conf
from avalanche.logging import InteractiveLogger, CSVLogger
from optimizers_lib import badam, bgd
from badam import BAdam
from vcl import VCL

import pandas as pd

import optuna
import wandb

"""
Select which experiment or method you would like
"""
exps = ['splitmnist', 'pmnist', 'splitfmnist']#'',,,'
methods = ['EWC','SI', 'VCL', 'BAdam','BGD', 'Naive', 'MAS']


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
"""
Good Method Parameters
"""
badam_etas = {'splitmnist': 0.01, 'splitfmnist': 0.1, 'pmnist': 0.1}
bgd_etas = {'splitmnist': 1.0, 'splitfmnist': 1.0, 'pmnist': 1.0}
badam_stds = {'splitmnist': 0.011, 'splitfmnist': 0.005,'pmnist': 0.01}
bgd_stds = {'splitmnist': 0.01, 'splitfmnist': 0.01, 'pmnist': 0.05}
mas_lambdas = {'splitmnist': 1.0, 'splitfmnist': 0.1}
vcl_epochs = {'pmnist': 30,'splitmnist': 20, 'splitfmnist': 20}



for exp in exps:    
    for run in range(1,26):
        for idx,method in enumerate(methods):       
            wandb.init(project=f"NEW_{exp}", name=f'{method}_{run}')
            random.seed(run)
            np.random.seed(run)
            torch.manual_seed(run)
            kwargs = get_exp_conf(exp)        
            csv_logger = CSVLogger(log_folder=f'results/{method}_{exp}_run_{run}')
            interactive_logger = InteractiveLogger()
            eval_plugin = EvaluationPlugin(
                accuracy_metrics(
                    minibatch=False, epoch=True, experience=True, stream=True
                ),
                forgetting_metrics(experience=True),
                loggers=[interactive_logger, csv_logger],
            )


            # CL Benchmark Creation
            train_stream = kwargs['dataset'].train_stream
            test_stream = kwargs['dataset'].test_stream

            # Prepare for training & testing
            optimizer = SGD(kwargs['model'].parameters(), lr=0.01)#
            if method == 'BAdam':
                optimizer = badam(kwargs['model'], mean_eta=badam_etas[exp], std_init=badam_stds[exp])
            elif method=='BGD':
                optimizer = bgd(kwargs['model'], mean_eta=bgd_etas[exp], std_init=bgd_stds[exp])
            elif method=='VCL':
                optimizer = SGD(kwargs['vcl_model'].parameters(), lr=0.01)#

            criterion = CrossEntropyLoss()

            replay = ReplayPlugin(mem_size=kwargs['coreset'])

            method_objects = {
                    'BAdam': BAdam(kwargs['model'], optimizer, criterion,\
                    train_mb_size=kwargs['batch_size'], train_epochs=20, eval_mb_size=kwargs['batch_size'], device=device, evaluator=eval_plugin),
                    'BGD': BAdam(kwargs['model'], optimizer, criterion,\
                    train_mb_size=kwargs['batch_size'], train_epochs=kwargs['epochs'], eval_mb_size=kwargs['batch_size'], device=device, evaluator=eval_plugin),                    
                    'EWC': EWC(kwargs['model'],optimizer, criterion,ewc_lambda=10, mode='separate',train_mb_size=kwargs['batch_size'], \
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
            
            cl_strategy = method_objects[method]
            # train and test loop over the stream of experiences
            results = []
            for train_exp in train_stream:
                mean_stds = cl_strategy.train(train_exp)
                evals = cl_strategy.eval(test_stream)#['Top1_Acc_Stream/eval_phase/test_stream/Task000']
                print(evals)
                results.append(evals['Top1_Acc_Stream/eval_phase/test_stream/Task000'])
            if isinstance(mean_stds, pd.DataFrame):
                mean_stds.to_csv(f"{method}_{run}_param_changes.csv", index=False)            
                mean, std = mean_stds['mean'].to_list(), mean_stds['std'].to_list()

            else:
                mean, std = None, None

            wandb.log(
                {
                    'Test Accuracy': results,
                    'Mean Per Epoch':mean, 
                    'Std Per Epoch': std
                }
            )
            wandb.finish()
