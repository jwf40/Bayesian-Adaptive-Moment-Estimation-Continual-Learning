import torch
import torch.nn as nn

class BaseNN(nn.Module):
    def __init__(self, n_samples):
        super(BaseNN,self).__init__()
        
        self.prior, self.posterior = None,None
        self.n_samples = n_samples

    def forward(self,x):
        pass

    def mean_prediction_probs(self, x):
        preds = []
        for _ in range(self.n_samples):
            preds.append(nn.Softmax(dim=1)(self.forward(x)))
        
        return preds.mean(0)

    def vcl_loss(self):
        pass

    def _calculate_kl_divergence(self):
        pass

    def _log_prob(self):
        pass

    def _init_variables(self):
        pass