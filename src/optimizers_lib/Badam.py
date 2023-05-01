import torch
import math
from torch.optim.optimizer import Optimizer


class Badam(Optimizer):
    """Implements BGD.
    A simple usage of BGD would be:
    for samples, labels in batches:
        for mc_iter in range(mc_iters):
            optimizer.randomize_weights()
            output = model.forward(samples)
            loss = cirterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.aggregate_grads()
        optimizer.step()
    """
    def __init__(self, params, std_init, mean_eta=1, mc_iters=10, betas=(0.9,0.999)):
        """
        Initialization of BGD optimizer
        group["mean_param"] is the learned mean.
        group["std_param"] is the learned STD.
        :param params: List of model parameters
        :param std_init: Initialization value for STD parameter
        :param mean_eta: Eta value
        :param mc_iters: Number of Monte Carlo iteration. Used for correctness check.
                         Use None to disable the check.
        """
        super(Fast_BGD, self).__init__(params, defaults={})
        assert mc_iters is None or (type(mc_iters) == int and mc_iters > 0), "mc_iters should be positive int or None."
        self.std_init = std_init
        self.mean_eta = mean_eta
        self.mc_iters = mc_iters
        self.betas = betas
        self.fast_eps = 1e-10
        self.n_steps = 1
        # Initialize mu (mean_param) and sigma (std_param)

        for group in self.param_groups:

            assert len(group["params"]) == 1, "BGD optimizer does not support multiple params in a group"
            # group['params'][0] is the weights
            assert isinstance(group["params"][0], torch.Tensor), "BGD expect param to be a tensor"
            # We use the initialization of weights to initialize the mean.
            group["mean_param"] = group["params"][0].data.clone()
            group["std_param"] = torch.zeros_like(group["params"][0].data).add_(self.std_init)
            group["mom"] = torch.zeros_like(group["params"][0].data)
            group["mom_var"] = torch.zeros_like(group["params"][0].data)
            group["std_mom"] = torch.zeros_like(group["params"][0].data)
            group["std_mom_var"] = torch.zeros_like(group["params"][0].data)
            group["lr"] = self.mean_eta
        self._init_accumulators()

    def get_mc_iters(self):
        return self.mc_iters

    def _init_accumulators(self):
        self.mc_iters_taken = 0
        for group in self.param_groups:
            group["eps"] = None
            group["grad_mul_eps_sum"] = torch.zeros_like(group["params"][0].data)
            group["grad_sum"] = torch.zeros_like(group["params"][0].data)

    def randomize_weights(self, force_std=-1):
        """
        Randomize the weights according to N(mean, std).
        :param force_std: If force_std>=0 then force_std is used for STD instead of the learned STD.
        :return: None
        """
        std_mean = 0
        for group in self.param_groups:
            mean = group["mean_param"]
            std = group["std_param"]
            std_mean += std.mean()
            if force_std >= 0:
                std = std.mul(0).add(force_std)
            group["eps"] = torch.normal(torch.zeros_like(mean), 1)
            # Reparameterization trick (here we set the weights to their randomized value):
            group["params"][0].data.copy_(mean.add(std.mul(group["eps"])))
        #print(std_mean/len(self.param_groups))

    def aggregate_grads(self, batch_size):
        """
        Aggregates a single Monte Carlo iteration gradients. Used in step() for the expectations calculations.
        optimizer.zero_grad() should be used before calling .backward() once again.
        :param batch_size: BGD is using non-normalized gradients, but PyTorch gives normalized gradients.
                            Therefore, we multiply the gradients by the batch size.
        :return: None
        """
        self.mc_iters_taken += 1
        groups_cnt = 0
        device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
        for group in self.param_groups:
            for each in group.keys():
                if isinstance(group[each], torch.Tensor):
                    group[each] = group[each].to(device)
            if group["params"][0].grad is None:
                continue
            assert group["eps"] is not None, "Must randomize weights before using aggregate_grads"
            groups_cnt += 1
            grad = group["params"][0].grad.data.mul(batch_size)
            # group["grad_sum"] = group["grad_sum"].detach().cpu()
            # grad = grad.detach().cpu()
            group["grad_sum"].add_(grad)
            group["grad_mul_eps_sum"].add_(grad.mul(group["eps"]))
            group["eps"] = None
            
        assert groups_cnt > 0, "Called aggregate_grads, but all gradients were None. Make sure you called .backward()"

    def step(self, closure=None):
        """
        Updates the learned mean and STD.
        :return:
        """
        # Makes sure that self.mc_iters had been taken.
        assert self.mc_iters is None or self.mc_iters == self.mc_iters_taken, "MC iters is set to " \
                                                                              + str(self.mc_iters) \
                                                                              + ", but took " + \
                                                                              str(self.mc_iters_taken) + " MC iters"
        max_grads = []
        for group in self.param_groups:
            mean = group["mean_param"]
            std = group["std_param"]
            mom = group["mom"]
            mom_var = group["mom_var"]
            mean_eta = group["lr"]
            # Divide gradients by MC iters to get expectation
            e_grad = group["grad_sum"].div(self.mc_iters_taken)
            e_grad_eps = group["grad_mul_eps_sum"].div(self.mc_iters_taken)
            max_grads.append(e_grad.max().item())

            

            alpha = (math.sqrt(1-self.betas[1]**(self.n_steps))/(1-self.betas[0]**(self.n_steps)))*(mean_eta*std.pow(2))

            mom.copy_(self.betas[0]*mom + (1-self.betas[0])*(e_grad))
            mom_var.copy_(self.betas[1]*mom_var + (1-self.betas[1])*(e_grad).pow(2))
            mean.add_(-alpha*(
                mom / (torch.sqrt(mom_var)+self.fast_eps)
            ) 
            )
            # tmp_mean = mean + (-alpha*std.pow(2)*(
            #     mom / (torch.sqrt(mom_var)+self.fast_eps)
            # ) 
            # )

            # mean.copy_(mean.mul(self.betas[1]) + tmp_mean.mul(1-self.betas[1]))
            # mean.copy_(mean/(1-(self.betas[1]**self.n_steps)))
            # std_mom.copy_(self.betas[0]*std_mom + (1-self.betas[0])*(e_grad_eps))
            # std_mom_var.copy_(self.betas[1]*std_mom + (1-self.betas[1])*(e_grad_eps.pow(2)))

            # sqrt_term = torch.sqrt(e_grad_eps.mul(std).div(2).pow(2).add(1)).mul(std)
            # std.copy_(sqrt_term.add(-alpha*0.5*std.pow(2)*(
            #    mom / (torch.sqrt(mom_var)+self.fast_eps))))
            
            sqrt_term = torch.sqrt(e_grad_eps.mul(std).div(2).pow(2).add(1)).mul(std)
            std.copy_(sqrt_term.add(-e_grad_eps.mul(std.pow(2)).div(2)))
            # sqrt_term = torch.sqrt(e_grad_eps.mul(std).div(2).pow(2).add(1)).mul(std)
            # std.copy_(sqrt_term.add(-0.5*std.pow(2)*alpha*(
            #     std_mom / (torch.sqrt(std_mom_var)+self.fast_eps)
            # )))
        self.n_steps+=1
        self.randomize_weights(force_std=0)
        self._init_accumulators()
