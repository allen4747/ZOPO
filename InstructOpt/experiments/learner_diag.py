import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
from misc import set_all_seed

class Network(nn.Module):
    def __init__(self, dim, hidden_size=100, depth=1, init_params=None):
        super(Network, self).__init__()
        self.activate = nn.ReLU()
        self.layer_list = nn.ModuleList()
        self.layer_list.append(nn.Linear(dim, hidden_size))
        for i in range(depth-1):
            self.layer_list.append(nn.Linear(hidden_size, hidden_size))
        self.layer_list.append(nn.Linear(hidden_size, 1))

        if init_params is None:
            # use initialization
            for i in range(len(self.layer_list)):
                set_all_seed(0)
                torch.nn.init.normal_(
                    self.layer_list[i].weight, mean=0, std=1.0)
                torch.nn.init.normal_(self.layer_list[i].bias, mean=0, std=1.0)
        else:
            # manually set the initialization vector
            for i in range(len(self.layer_list)):
                self.layer_list[i].weight.data = init_params[i*2]
                self.layer_list[i].bias.data = init_params[i*2+1]

    def forward(self, x):
        y = x
        for i in range(len(self.layer_list)-1):
            y = self.activate(self.layer_list[i](y))
        y = self.layer_list[-1](y)
        return y
        
        
def empirical_ntk(network, params, x1, x2, compute='full'):
    # Compute J(x1)
    jac1 = torch.vmap(torch.func.jacrev(network), (None, 0))(params, x1)
    jac1 = [jac1[j].flatten(2) for j in jac1]
    # jac1 = [j.flatten(2) for j in jac1]
    
    # Compute J(x2)
    jac2 = torch.vmap(torch.func.jacrev(network), (None, 0))(params, x2)
    jac2 = [jac2[j].flatten(2) for j in jac2]
    # jac2 = [j.flatten(2) for j in jac2]
    
    # Compute J(x1) @ J(x2).T
    einsum_expr = None
    if compute == 'full':
        einsum_expr = 'Naf,Mbf->NMab'
    elif compute == 'trace':
        einsum_expr = 'Naf,Maf->NM'
    elif compute == 'diagonal':
        einsum_expr = 'Naf,Maf->NMa'
    else:
        assert False
        
    # ntk_x1 = torch.stack([torch.einsum(einsum_expr, j1, j1) for j1 in jac1])
    # ntk_x1 = ntk_x1.sum(0)
    
    ntk_x2x1 = torch.stack([torch.einsum(einsum_expr, j2, j1) for j1, j2 in zip(jac1, jac2)])
    ntk_x2x1 = ntk_x2x1.sum(0)
    return ntk_x2x1
    
    
class GP_NTK():
    def __init__(self, network) -> None:
        self.net = network
        # fnet, self.params = make_functional(self.net)
        # self.fnet = lambda params, x: fnet(params, x.unsqueeze(0)).squeeze(0)
        self.params = dict(self.net.named_parameters())
        self.fnet = lambda params, x: torch.func.functional_call(self.net, params, (x.unsqueeze(0),)).squeeze(0)

    def fit(self, inputs, outputs, sigma=0.01, ntk_func=empirical_ntk):
        # inputs: queried inputs for GP fit with shape (batch, input_dim)
        # outputs: the function value of queried inputs with shape (batch, 1)
        self.inputs = inputs
        self.outputs = outputs.reshape(-1,1)
        n_input = inputs.size(0)
        
        ntk = ntk_func(self.fnet, self.params, inputs, inputs).squeeze((2,3))
        try:
            self.K_inv = torch.linalg.inv(ntk +  sigma**2 * torch.eye(n_input).cuda())
        except:
            self.K_inv = torch.linalg.inv(ntk +  10 * torch.eye(n_input).cuda())
        # print(ntk.shape, self.K_inv.shape)
        return
    
    def pred(self, input_eval, ntk_func=empirical_ntk, get_var=False):
        # input_eval: point to get prediction with shape (1, input_dim)
        ntk_mix = ntk_func(self.fnet, self.params, self.inputs, input_eval).squeeze((2,3))
        mean = torch.mm(torch.mm(ntk_mix, self.K_inv), self.outputs)
        
        if get_var:
            kxx = ntk_func(self.fnet, self.params, input_eval, input_eval).squeeze()
            var = kxx - torch.mm(torch.mm(ntk_mix, self.K_inv), ntk_mix.T)
            return mean, var
        else:
            return mean
    
    def pred_var(self, input_eval, ntk_func=empirical_ntk):
        ntk_mix = ntk_func(self.fnet, self.params, self.inputs, input_eval).squeeze((2,3))
        kxx = ntk_func(self.fnet, self.params, input_eval, input_eval).squeeze()
        var = kxx - torch.mm(torch.mm(ntk_mix, self.K_inv), ntk_mix.T)
        return var
    
    
# to test the functionality of GP_NTK
if __name__ == "__main__":
    batch, input_dim = 10, 5
    torch.manual_seed(0)
    inputs = torch.randn(batch, input_dim).cuda()
    outputs = torch.randn(batch).cuda()
    input_eval = torch.randn(1, input_dim).cuda()
    input_eval.requires_grad_()
    func = Network(dim=input_dim)
    func.cuda()
    gp_ntk = GP_NTK(network = func)
    gp_ntk.fit(inputs, outputs)
    mean, var = gp_ntk.pred(input_eval, get_var=True)
    # var = gp_ntk.pred_var(input_eval)
    print("pred mean:", mean, "pred_var:", var)
    gradient = torch.autograd.grad(mean, input_eval)
    print("Input gradient:", gradient)
    
    
