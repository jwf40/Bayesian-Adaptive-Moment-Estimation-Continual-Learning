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

# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
exps = ['pmnist']#'',,,'
methods = ['BAdam']#,'VCL']#['BAdam']#,'', #, 'BGD''VCL''BAdam''BGD','EWC','Naive', 'MAS','SI'

badam_etas = {'splitmnist': 0.01, 'splitfmnist': 0.1, 'pmnist': 0.1}
bgd_etas = {'splitmnist': 1.0, 'splitfmnist': 1.0, 'pmnist': 1.0}
badam_stds = {'splitmnist': 0.011, 'splitfmnist': 0.005,'pmnist': 0.01}
bgd_stds = {'splitmnist': 0.01, 'splitfmnist': 0.01, 'pmnist': 0.05}

mas_lambdas = {'splitmnist': 1.0, 'splitfmnist': 0.1}
vcl_epochs = {'pmnist': 100,'splitmnist': 120, 'splitfmnist': 120}

etas = [0.01]#,0.05,0.075,0.1, 0.2]
stds = [0.011]#, 0.017, 0.02, 0.03, 0.04]

for exp in exps:
    for run in range(1, 11):
        #for eta,std in zipped:
        # for std in stds:
        #     for eta in etas:
                for idx,method in enumerate(methods):       
                    if method == 'VCL' and exp == 'splitmnist':
                         continue
                    random.seed(run)
                    np.random.seed(run)
                    torch.manual_seed(run)
                    kwargs = get_exp_conf(exp)        
                    csv_logger = CSVLogger(log_folder=f'results/no_corset/NEWVAL{method}_{exp}_run_{run}')
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
                        optimizer = Adam(kwargs['model'].parameters(), lr=0.001)#

                    criterion = CrossEntropyLoss()

                    replay = ReplayPlugin(mem_size=kwargs['coreset'])

                    method_objects = [ #kwargs['epochs']
                            # BAdam(kwargs['model'], optimizer, criterion,\
                            # train_mb_size=kwargs['batch_size'], train_epochs=20, eval_mb_size=kwargs['batch_size'], device=device, evaluator=eval_plugin)]
                            BAdam(kwargs['model'], optimizer, criterion,\
                            train_mb_size=kwargs['batch_size'], train_epochs=kwargs['epochs'], eval_mb_size=kwargs['batch_size'], device=device, evaluator=eval_plugin),                    
                            EWC(kwargs['model'],optimizer, criterion,ewc_lambda=10, mode='online',decay_factor=0.9,train_mb_size=kwargs['batch_size'], \
                                train_epochs=kwargs['epochs'], eval_mb_size=kwargs['batch_size'], device=device, evaluator=eval_plugin),
                            Naive(kwargs['model'], optimizer, criterion, \
                                 train_mb_size=kwargs['batch_size'], train_epochs=kwargs['epochs'], eval_mb_size=kwargs['batch_size'], device=device, evaluator=eval_plugin),
                            MAS(kwargs['model'], optimizer, criterion, lambda_reg=0.1,train_mb_size=kwargs['batch_size'], \
                                train_epochs=kwargs['epochs'], eval_mb_size=kwargs['batch_size'], device=device, evaluator=eval_plugin),                
                            SynapticIntelligence(kwargs['model'],optimizer, criterion,si_lambda=1.0, train_mb_size=kwargs['batch_size'], \
                                train_epochs=kwargs['epochs'], eval_mb_size=kwargs['batch_size'], device=device, evaluator=eval_plugin),
                            # VCL(kwargs['vcl_model'],optimizer, criterion,train_mb_size=kwargs['batch_size'], \
                            #     train_epochs=vcl_epochs[exp], eval_mb_size=kwargs['batch_size'], device=device, evaluator=eval_plugin)
                    ]
                    #plugins=[replay]
                    # Naive(kwargs['model'], optimizer, criterion, plugins=[replay, SpectralForgettingPlugin(lambda_reg=1.0,high_or_low='high', sigma=5.0)],\
                    #         train_mb_size=kwargs['batch_size'], train_epochs=kwargs['epochs'], eval_mb_size=kwargs['batch_size'], device=device, evaluator=eval_plugin),

                    cl_strategy = method_objects[idx]
                    # train and test loop over the stream of experiences
                    results = []
                    for train_exp in train_stream:
                        cl_strategy.train(train_exp)
                        results.append(cl_strategy.eval(test_stream))

# Continual learning strategy

