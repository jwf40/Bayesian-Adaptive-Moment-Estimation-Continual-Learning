import torch
import torch.nn as nn
import torchbnn as bnn
from .base_nn import BaseNN

class MLP(BaseNN):
    def __init__(self, n_samples=50):    
        super(MLP,self).__init__(n_samples)
        self.model = nn.Sequential(bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=28*28, out_features=1024),\
                                    nn.ReLU(),
                                    bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=1024, out_features=1024),
                                    nn.ReLU(),
                                    bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=1024, out_features=10))

    def forward(self,x):
        x = self.model(x)
        return x

    def vcl_loss(self):
        return self._log_prob
    
    def update_probs(self):
        for m in self.model:
            if isinstance(m,bnn.BayesLinear):
                m.prior_mu = self.posterior

    def _log_prob
        