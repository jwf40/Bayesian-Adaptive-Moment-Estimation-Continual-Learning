import torch.nn as nn
import torch.nn.functional as F

from models.bbblinear import BBBLinear
from avalanche.models.base_model import BaseModel

class VCLModel(nn.Module, BaseModel):

    def __init__(self, num_classes=10,
        input_size=28 * 28,
        hidden_size=512,
        hidden_layers=1):
        """
        input: 1 x 28 x 28
        output: 1 classifiers with 10 nodes
        hidden: [100, 100]
        """
        self.n_classes = num_classes
        super().__init__()
        layers = [
                BBBLinear(input_size, hidden_size),
        ]
        for layer_idx in range(hidden_layers - 1):
            layers.append(                
                        BBBLinear(hidden_size, hidden_size),                   
                    )
                

        self.features = layers
        self.classifier = BBBLinear(hidden_size, num_classes)
        self._input_size = input_size

    def forward(self, x):
        out = x.view(-1, 784)
        for layer in self.features:
            out = F.relu(layer(out))
        return self.classifier(out)
    
    def get_features(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)

        for layer in self.features:
            x = layer(x)
        return x

    def get_kl(self):
        kl = 0.0
        for layer in self.features:
            kl += layer.kl_loss()
        kl += self.classifier.kl_loss()
        return kl

    def update_prior(self):
        for layer in self.features:
            layer.prior_W_mu = layer.W_mu.data
            layer.prior_W_sigma = layer.W_sigma.data
            if layer.use_bias:
                layer.prior_bias_mu = layer.bias_mu.data
                layer.prior_bias_sigma = layer.bias_sigma.data

        self.classifier.prior_W_mu = self.classifier.W_mu.data
        self.classifier.prior_W_sigma = self.classifier.W_sigma.data
        if self.classifier.use_bias:
            self.classifier.prior_bias_mu = self.classifier.bias_mu.data
            self.classifier.prior_bias_sigma = self.classifier.bias_sigma
    def _to(self, device):
        self.to(device)
        for layer in self.features:
            layer.to(device)