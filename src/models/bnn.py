import torch
import torch.nn as nn
import torchbnn as bnn


class MLP(nn.Module):
    def __init__(self, n_samples=50, n_tasks=3):    
        super(MLP,self).__init__()
        print("WARNING, CURRENTLY MM HEAD IS NOT BAYESIAN")
        self.n_samples = n_samples

        #Main feature extractor
        self.model = nn.Sequential(bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=28*28, out_features=100),\
                                    nn.ReLU(),
                                    bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=100, out_features=100),
                                    nn.ReLU())

        #Head to solve tasks
        self.prediction_head = nn.Sequential(bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=100, out_features=10))
        
        #Head to matchmake
        self.matchmaking_head = nn.Sequential(nn.Linear(in_features=100+(28*28), out_features=100),
                                nn.Linear(in_features=100, out_features=n_tasks))
        
        #To be set after initial training
        self.entropy_threshold = 0.0
    
    def forward(self,x):
        return self.prediction_head(self.model(x))

    def is_confident(self,entropy):
        return entropy <= self.entropy_threshold

    def prediction_means(self,x):
        means = []
        for _ in range(self.n_samples):
            means.append(nn.Softmax(dim=1)(self.forward(x)))
        return torch.stack(means).mean(dim=0)        

    def predictive_entropy(self,x):
        means = self.prediction_means(x)
        return self.predictive_entropy_from_means(means)

    def predictive_entropy_from_means(self,means):
        eps = 1e-3
        entropy = means*torch.log(torch.add(means,eps))
        return -entropy.sum(dim=1)
    
    def set_grads(self, module,requires_grad=True):
        for child in module.children():
            for param in child.parameters():
                param.requires_grad = requires_grad

    def matchmake_loss(self,pred,y):
        loss = nn.CrossEntropyLoss()
        print(pred, torch.Tensor([y]))
        return loss(pred.cpu(),torch.Tensor([y]))
    
    def choose_teacher(self, x):
        #Get mean prediction, return selection
        means = []
        for _ in range(self.n_samples):
            tmp_ = torch.cat((x,self.model(x)),dim=1)
            predictions = self.matchmaking_head(tmp_)
            means.append(nn.Softmax()(predictions))
        means = torch.stack(means)
        return means.mean(dim=0)
    
    def prediction_params(self):
        return list(self.model.named_parameters()) + list(self.prediction_head.named_parameters())

    def matchmaking_layers(self):
        return [{'params': params} for l, (name, params) in enumerate(self.matchmaking_head.named_parameters())]
    def prediction_layers(self):
        return [{'params': params} for l, (name, params) in enumerate(list(self.model.named_parameters())+list(self.prediction_head.named_parameters()))]