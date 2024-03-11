import torch as t
import torch.nn as nn
import torchvision as tv, torchvision.transforms as tr
import tqdm
import matplotlib.pyplot as plt
import math
from torch.nn.utils.parametrizations import spectral_norm
import pathlib
import os
import wandb
import numpy as np
from model.energy import ConvNetwork, ConcatEnergyBase
from model.utils import target_function, feedback_loss
from model.utils.plot_utils import plot, plot_graph
from model.bias import BiasExplicit, init_bias
from model.sample import sample_p_d, sample_LangevinDynamics, noise
from model.proposal import getter_proposal

from backpack import extend

import time

t.cuda.empty_cache()

import torch
from torch.utils.data import Dataset

class PlaneDataset(Dataset):
    def __init__(self, num_points, flip_axes=False):
        self.num_points = num_points
        self.flip_axes = flip_axes
        self.data = None
        self.reset()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.num_points

    def reset(self):
        self._create_data()
        if self.flip_axes:
            x1 = self.data[:, 0]
            x2 = self.data[:, 1]
            self.data = torch.stack([x2, x1]).t()

    def _create_data(self):
        raise NotImplementedError
    
class FourCirclesDataset(PlaneDataset):
    def __init__(self, num_points, flip_axes=False):
        if num_points % 4 != 0:
            raise ValueError('Number of data points must be a multiple of four')
        super().__init__(num_points, flip_axes)

    @staticmethod
    def create_circle(num_per_circle, std=0.1):
        u = torch.rand(num_per_circle)
        x1 = torch.cos(2 * np.pi * u)
        x2 = torch.sin(2 * np.pi * u)
        data = 2 * torch.stack((x1, x2)).t()
        data += std * torch.randn(data.shape)
        return data

    def _create_data(self):
        num_per_circle = self.num_points // 4
        centers = [
            [-1, -1],
            [-1, 1],
            [1, -1],
            [1, 1]
        ]
        self.data = torch.cat(
            [self.create_circle(num_per_circle) - torch.Tensor(center)
             for center in centers]
        )


def forward_pass(sample_data, sample_q, energy, K, m,):
    x_p_d = sample_data(m=m)
    x_q, inv_log_weights, dic_sample = sample_q(K, m)  
    energy_data = energy(x_p_d)
    energy_q = energy(x_q)
    log_z = t.logsumexp(-energy_q-inv_log_weights-math.log(m), dim=0)
    log_ESS = 2*t.logsumexp(-energy_q-inv_log_weights, dim=0)- t.logsumexp(-2*energy_q-2*inv_log_weights.detach(), dim=0)
    ESS = t.exp(log_ESS)
    log_likelihood = -energy_data.mean() - log_z
    snl = -energy_data.mean() - (log_z).exp() +1  

    dic_forward = {
        "x_data": x_p_d,
        "x_q": x_q,
        "inv_log_weights": inv_log_weights,
        "log_z": log_z,
        "log_likelihood": log_likelihood,
        "snl": snl,
        "ESS": ESS,  
        "energy_data": energy_data,
        "energy_q": energy_q,
        }

    return dic_forward, dic_sample


import argparse

def parser_default():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annealing", default="noannealing", type=str, choices=["annealing", "cosine_annealing", "noannealing"])
    parser.add_argument("--clamp", action="store_true", help="clamp")
    parser.add_argument("--no_sn", action="store_true", help="sn")
    parser.add_argument("--K", default=100, type=int, help="K")
    parser.add_argument("--n_i", default=2000, type=int, help="n_i")
    parser.add_argument("--lambda_reg", default=0.1, type=float, help="lambda_reg")
    parser.add_argument("--n_f", default=64, type=int, help="n_f")
    parser.add_argument("--m", default=8**2, type=int, help="m")
    parser.add_argument("--n_ch", default=1, type=int, help="n_ch")
    parser.add_argument("--im_sz", default=32, type=int, help="im_sz")
    parser.add_argument("--sigma_data", default=3e-2, type=float, help="sigma_data")
    parser.add_argument("--sigma_step", default=1e-2, type=float, help="sigma_step")
    parser.add_argument("--seed", default=1, type=int, help="seed")
    parser.add_argument("--log_method", help="Use log for SNL or LogBound", action="store_true")
    parser.add_argument("--step_size", default=1.0, type=float, help="step_size")
    parser.add_argument("--method_kick_in", default=0.5, type=float, help="When to switch to snl or log method")
    parser.add_argument("--normalize_step", default=0, type=int, help="When to switch to snl or log method")

    parser.add_argument("--variance_eval", action="store_true", help="variance_eval")


    parser.add_argument("--proposal_name", default="uniform", type=str, choices=["gaussian", "uniform"])
    parser.add_argument("--base_name", default=None, type=str, choices=["gaussian", "uniform",])
    parser.add_argument("--use_proposal_in_sample", action="store_true", help="use_proposal_in_sample")


    return parser

# sn = True

if __name__ == "__main__":

    parser = parser_default()
    args = parser.parse_args()

    annealing = args.annealing
    clamp = args.clamp
    sn = not args.no_sn
    K = args.K
    n_i = args.n_i
    lambda_reg = args.lambda_reg
    n_f = args.n_f
    m = args.m
    n_ch = args.n_ch
    im_sz = args.im_sz
    sigma_data = args.sigma_data
    sigma_step = args.sigma_step
    step_size = args.step_size
    seed = args.seed
    log_method = args.log_method
    method_kick_in = args.method_kick_in
    normalize_step = args.normalize_step

    t.cuda.empty_cache()
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)


    folder = "standardenergy_{}_clamp{}_{}_sn_{}_{}".format(annealing, clamp, K, sn, time.strftime("%Y%m%d-%H%M%S"))
    p = pathlib.Path(folder)
    p.mkdir(parents=True, exist_ok=True)
    logger = wandb.init(project="short_term_analysis_snl", config=vars(args), name=folder)

    try :
        device = t.device('cuda' if t.cuda.is_available() else 'cpu')

        # Data
        transform = tr.Compose([tr.Resize(im_sz), tr.ToTensor(),])
        # p_d = t.stack([x[0] for x in tv.datasets.CIFAR10(root='data/cifar10', download=True, transform=transform)]).to(device)
        p_d = t.stack([x[0] for x in tv.datasets.MNIST(root='data/mnist', download=True, transform=transform,)]).to(device)


        # Split p_d in trian adn val randomly
        randint = t.randperm(p_d.shape[0])
        p_d_train = p_d[randint[:50000]]
        p_d_val = p_d[randint[50000:]]
        
        # Sampling data and proposal
        noise_function = lambda x: noise(x, sigma_data=sigma_data)
        p_d_val = noise(p_d_val, sigma_data=sigma_data)
        current_proposal = getter_proposal(args.proposal_name, p_d_train+0.1*t.randn_like(p_d_train),).to(device)
        current_base = getter_proposal(args.base_name, p_d_train+0.1*t.randn_like(p_d_train), )

        # Network
        fnn = ConvNetwork(n_c=n_ch, n_f=n_f, sn=sn)
        bias = BiasExplicit().to(device)

        energy = ConcatEnergyBase(fnn, base=current_base, bias=bias).to(device)
        current_target = lambda k, K, x : target_function(k, K, x, annealing=annealing, energy=energy, proposal=current_proposal, use_proposal=args.use_proposal_in_sample,)

        sample_data = lambda m: sample_p_d(m, p_d_train, noise_function=noise_function)
        dataset_val = t.utils.data.TensorDataset(p_d_val)
        dataloader_val = t.utils.data.DataLoader(dataset_val, batch_size=m, shuffle=True)
        sample_q = lambda K,m : sample_LangevinDynamics(
                                                    K = K,
                                                    m = m,
                                                    target_function=current_target,
                                                    step_size=step_size,
                                                    sigma_step=sigma_step,
                                                    clamp=clamp,
                                                    annealing=annealing,
                                                    device=device,
                                                    proposal=current_proposal,
                                                    )
        
        # Optim model
        optim = t.optim.Adam(energy.parameters(), lr=1e-4, betas=[.9, .999], )
        optim_bias = t.optim.SGD(bias.parameters(), lr=1e-2, momentum=0.1)
        bias.explicit_bias.data +=init_bias(energy, sample_q, m, K, 10,).data

        for i in tqdm.tqdm(range(n_i)):
            energy.train()

            if args.variance_eval :
                grad_data = []
                grad_snl = []
                grad_log = []
                grad_b = []
                for k in range(10):
                    dic_forward, dic = forward_pass(sample_data, sample_q, energy, K, m,)
                    loss_data = dic_forward["energy_data"].mean()
                    loss_data.backward(retain_graph=True)
                    grad_data.append(torch.cat([param.grad.clone().reshape(-1,1) for param in fnn.parameters()]))
                    optim.zero_grad()

                    loss_z_snl = dic_forward["log_z"].exp()-1
                    loss_z_snl.backward(retain_graph=True)
                    grad_snl.append(torch.cat([param.grad.clone().reshape(-1,1) for param in fnn.parameters()]))
                    grad_b.append(bias.explicit_bias.grad.clone().reshape(-1,1))
                    optim.zero_grad()

                    loss_z_log = dic_forward["log_z"]
                    loss_z_log.backward(retain_graph=True)
                    grad_log.append(torch.cat([param.grad.clone().reshape(-1,1) for param in fnn.parameters()]))
                    optim.zero_grad()

                    
                grad_data = torch.cat(grad_data, dim=1)
                grad_snl = torch.cat(grad_snl, dim=1)
                grad_log = torch.cat(grad_log, dim=1)
                grad_b = torch.cat(grad_b, dim=1)
                # print(grad_b)
                # print(grad_data.var(dim=0).shape)
                # assert grad_data.norm(dim=0).shape == 
                logger.log({
                    "l2/grad_data_l2_mean": grad_data.norm(dim=0).mean().log().item(),
                    "var/grad_data_variance_mean": grad_data.var(dim=0).mean().log().item(),
                    "l2/grad_snl_l2_mean": grad_snl.norm(dim=0).mean().log().item(),
                    "var/grad_snl_variance_mean": grad_snl.var(dim=0).mean().log().item(),
                    "l2/grad_log_l2_mean": grad_log.norm(dim=0).mean().log().item(),
                    "var/grad_log_variance_mean": grad_log.var(dim=0).mean().log().item(),
                    "l2/grad_b_l2_mean": grad_b.norm().log().item(),
                    "var/grad_b_variance_mean": grad_b.var().log().item(),
                }, step=i)

                
            
            dic_forward, dic = forward_pass(sample_data, sample_q, energy, K, m,)



            x_q = dic_forward["x_q"]
            x_p_d = dic_forward["x_data"]
            log_z = dic_forward["log_z"]
            log_likelihood = dic_forward["log_likelihood"]
            snl = dic_forward["snl"]
            ESS = dic_forward["ESS"]
            inv_log_weights = dic_forward["inv_log_weights"]

            if i< n_i*method_kick_in:
                loss = (energy(x_p_d).mean() - energy(x_q).mean()) 
            else :
                if log_method :
                    loss = -log_likelihood
                else :
                    loss = -snl

            optim.zero_grad()
            (loss).backward(retain_graph=True)
            for param in energy.parameters():
                if param.grad is not None:
                    param.grad = param.grad.clamp(-1,1)
            if log_method :
                optim_bias.zero_grad()
                bias.explicit_bias.grad = -((log_z).exp().flatten() -1).clamp(-1,1)
            else :
                bias.explicit_bias.grad =bias.explicit_bias.grad.clamp(-1,1)
            optim_bias.step()
            optim.step()

            if i%100 == 0:
                energy.eval()
                log_z = init_bias(energy, sample_q, m, K, 100,)
                total_log_likelihood = 0
                energy_val = 0
                nb_element = 0
                for x_val in iter(dataloader_val):
                    x_val = x_val[0].to(device)
                    batch_size = x_val[0].shape[0]
                    energy_val = (energy_val*nb_element + bias(energy(x_val)).mean() * batch_size)/(nb_element+batch_size)
                log_likelihood = - energy_val - log_z
                snl = -energy_val - (log_z).exp() +1
                logger.log({"val/log_likelihood": log_likelihood.item(),
                        "val/snl": snl.item(),
                        "val/energy": energy_val.item(),
                        "val/f_theta": fnn(x_val).mean().item(),
                        "val/log_z":log_z.item()},
                step=i)
                    




            logger.log({'E(x_p_d)': fnn(x_p_d).mean().item(), 'E(x_q)': fnn(x_q).mean().item(), 'step': i}, step=i)
            if len(dic['f_prime_mean']):
                    logger.log({
                    'grad_norm_max': np.max(dic['f_prime_mean']),
                    'grad_norm_mean': np.mean(dic['f_prime_mean']),
                    'grad_norm_std': np.std(dic['f_prime_mean']),
                    'grad_norm_min':np.min(dic['f_prime_mean']),
                    }, step=i)
            logger.log({
                    'bias/bias': bias.explicit_bias.item(),
                    'loss': loss.item(),
                    'bias/log_z': log_z.item(),
                    'log_likelihood': log_likelihood.item(),
                    'snl' : snl.item(),
                    'inv_log_weights': inv_log_weights.mean().item(),
                    'bias/grad_bias': log_z.exp().item()-1,
                    'ESS': ESS.item(),
                }, step=i)



            if i % 50 == 0 :
                for key in dic.keys():
                    plot_graph(os.path.join(folder, '{}_{:>06d}.png'.format(key, i)), dic[key], None, logger, i, name=key)
                
            if i % 50 == 0:
                # print('{:>6d} E(x_p_d)={:>14.9f} E(x_q)={:>14.9f}'.format(i, energy(x_p_d).mean(), energy(x_q).mean()))
                plot(os.path.join(folder, 'x_q_{:>06d}.png'.format(i)), x_q, logger, i,name = "x_q")
                x_q, _, _ = sample_q(10, m,)
                plot(os.path.join(folder, 'x_q_100_{:>06d}.png'.format(i)), x_q, logger, i,name = "x_q_100")


        
        wandb.finish()
    except Exception as e:
        logger.log({"error": e})
        wandb.finish()
        raise e
            