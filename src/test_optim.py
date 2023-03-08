import torch
from itertools import chain
from torch.optim.optimizer import Optimizer, _use_grad_for_differentiable

class BAdam(Optimizer):
    def __init__(self, named_params, named_buffer,std_init=1, mean_eta=1):
        param_groups = [{'params': [], 'name':'weight_mu'}, {'params': [], 'name':'weight_log_sigma'}, {'params': [], 'name':'bias_mu'},\
             {'params': [], 'name':'bias_log_sigma'}, {'params':[], 'name':'weights'},{'params':[], 'name': 'biases'}]
        for n,p in named_params:
            for ele in param_groups:
                if ele['name'] in n:
                    ele['params'].append(p)
                    continue
                
        self._param_groups = param_groups
        self.named_buffer = named_buffer
        self.named_params = named_params
        super(BAdam, self).__init__(param_groups, defaults={})
        
        self.std_init = std_init
        self.mean_eta = mean_eta

        self.batch_size = None

    def set_batch_size(self,batch_size):
        self.batch_size = batch_size
    
    def step(self, closure=None):    
        assert self.batch_size != None, 'ERROR, YOU MUST SET BATCH_SIZE BEFORE CALLING STEP()'
        for each in self.param_groups:
            n = each['name']
            p = each['params']
            ns = n.split('.')
            weight_grads = list(filter(lambda x: x['name']=='.'.join(ns[:-1]) + '.weights'))[0].grad.data.mul(self.batch_size)
            bias_grads = self.named_params['.'.join(ns[:-1]) + '.biases'].grad.data.mul(self.batch_size)

            if 'weight_mu' in n:                
                sigma = self.named_params['.'.join(ns[:-1]) + '.weight_log_sigma']
                sigma = torch.exp(sigma)
                p.add_(- self.mean_eta*(sigma.pow(2))*weight_grads)
            elif 'weight_log_sigma' in n: 
                sqrt_term = torch.sqrt(weight_grads.mul(self.named_buffer['.'.join(ns[:-1]) + '.weight_eps']).mul(p).div(2).add(1)).mul(p)
                p.copy_(sqrt_term.add(-weight_grads.mul(self.named_buffer['.'.join(ns[:-1]) + '.weight_eps']).mul(p.pow(2)).div(2)))
            elif 'bias_mu' in n:
                sigma = self.named_params[''.join[ns[:-1]].append('bias_log_sigma')]
                sigma = torch.exp(sigma)
                p.add_(- self.mean_eta*(sigma.pow(2))*bias_grads)
            elif 'bias_log_sigma' in n:
                sqrt_term = torch.sqrt(bias_grads.mul(self.named_buffer['.'.join(ns[:-1]) + '.bias_eps']).mul(p).div(2).add(1)).mul(p)
                p.copy_(sqrt_term.add(-bias_grads.mul(self.named_buffer['.'.join(ns[:-1]) + '.bias_eps']).mul(p.pow(2)).div(2)))