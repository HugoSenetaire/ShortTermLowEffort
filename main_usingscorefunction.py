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
from model.energy import ConvNetwork
from model.utils import target_function
from model.utils.plot_utils import plot, plot_graph
from model.bias import BiasExplicit, init_bias
from model.sample import sample_p_d, sample_LangevinDynamicsScoreControl, sample_LangevinDynamicsFullTrajectory, noise, sample_base
from model.score_function import ScoreFunction
import time


t.cuda.empty_cache()



import argparse

def parser_default():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annealing", default="noannealing", type=str, choices=["annealing", "cosine_annealing", "noannealing"])
    parser.add_argument("--clamp", default="", type=str, choices=["clamp", ""])
    parser.add_argument("--sn", action="store_true", help="sn")
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
    parser.add_argument("--method_kick_in", default=0.5, type=float, help="When to switch to snl or log method")
    parser.add_argument("--normalize_step", default=0, type=int, help="When to switch to snl or log method")
    parser.add_argument("--step_size", default=1.0, type=float, help="step_size")


    parser.add_argument("--score_name", default="UNET", type=str, help="Name of the score function")
    parser.add_argument("--score_type_method", default="linear", type=str, help="Type of the score function", choices=["linear", "periodic"])
    parser.add_argument("--nb_optim_score", default=1, type=int, help="Number of optimization step for the score function")

    return parser

# sn = True

if __name__ == "__main__":

    parser = parser_default()
    args = parser.parse_args()

    annealing = args.annealing
    clamp = args.clamp
    sn = args.sn
    K = args.K
    n_i = args.n_i
    lambda_reg = args.lambda_reg
    n_f = args.n_f
    m = args.m
    n_ch = args.n_ch
    im_sz = args.im_sz
    sigma_data = args.sigma_data
    sigma_step = args.sigma_step
    step_size = 1.0
    seed = args.seed
    log_method = args.log_method
    method_kick_in = args.method_kick_in
    normalize_step = args.normalize_step

    t.cuda.empty_cache()
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)


    folder = "scorefunction_{}_{}_{}_sn_{}_{}".format(annealing, clamp, K, sn, time.strftime("%Y%m%d-%H%M%S"))
    p = pathlib.Path(folder)
    p.mkdir(parents=True, exist_ok=True)
    logger = wandb.init(project="short_term_analysis_snl", config=vars(args), name=folder)

    try :
        device = t.device('cuda' if t.cuda.is_available() else 'cpu')

        # Network
        f = ConvNetwork(n_c=n_ch, n_f=n_f, sn=sn).to(device)
        bias = BiasExplicit().to(device)

        # Data
        transform = tr.Compose([tr.Resize(im_sz), tr.ToTensor(),])
        # p_d = t.stack([x[0] for x in tv.datasets.CIFAR10(root='data/cifar10', download=True, transform=transform)]).to(device)
        p_d = t.stack([x[0] for x in tv.datasets.MNIST(root='data/mnist', download=True, transform=transform,)]).to(device)
        # Split p_d in trian adn val randomly
        randint = t.randperm(p_d.shape[0])
        p_d_train = p_d[randint[:50000]]
        p_d_val = p_d[randint[50000:]]

        score_function = ScoreFunction(model_name=args.score_name, K=K, input_nc=n_ch, output_nc=n_ch, img_size=im_sz, ngf=n_f, type_method=args.score_type_method).to(device)

        
        # Sampling data and proposal
        noise_function = lambda x: noise(x, sigma_data=0.1)
        current_target = lambda k, K, x : target_function(k, K, x, annealing=annealing, f=f)
        current_sample_base = lambda m : sample_base(m, n_ch=n_ch, im_sz=im_sz)

        sample_data = lambda m: sample_p_d(m, p_d_train, noise_function=noise_function)
        dataset_val = t.utils.data.TensorDataset(p_d_val)
        dataloader_val = t.utils.data.DataLoader(dataset_val, batch_size=m, shuffle=True)
        sample_q = lambda K,m : sample_LangevinDynamicsScoreControl(
                                                    K = K,
                                                    m = m,
                                                    target_function=current_target,
                                                    step_size=step_size,
                                                    sigma_step=sigma_step,
                                                    clamp=clamp,
                                                    annealing=annealing,
                                                    device=device,
                                                    sample_base=current_sample_base,
                                                    score_function=score_function,
                                                    )
        
        sample_trajectory = lambda K,m : sample_LangevinDynamicsFullTrajectory(
                                                    K = K,
                                                    m = m,
                                                    target_function=current_target,
                                                    step_size=step_size,
                                                    sigma_step=sigma_step,
                                                    clamp=clamp,
                                                    annealing=annealing,
                                                    device=device,
                                                    sample_base=current_sample_base,
        )


        # Optim model
        optim = t.optim.Adam(f.parameters(), lr=1e-4, betas=[.9, .999], )
        optim_score = t.optim.Adam(score_function.parameters(), lr=1e-4, betas=[.9, .999], )
        optim_bias = t.optim.SGD(bias.parameters(), lr=1e-1, momentum=0.5,)
        

        bias.explicit_bias.data = init_bias(bias, f, sample_q, m, K, 10,).data

        total_step = 0
        for i in tqdm.tqdm(range(n_i)):
            score_function.requires_grad_(True)
            for j in range(args.nb_optim_score):
                score_function.train()
                trajectory,  epsilon_trajectory, dic = sample_trajectory(K, m)
                all_trajectory = t.cat(trajectory, dim=0)
                all_k = t.cat([t.ones(m)*k for k in range(1,K+1)], dim=0).to(device)
                all_epsilon = t.cat(epsilon_trajectory, dim=0)
                optim_score.zero_grad()
                loss = (score_function(all_trajectory, all_k)-step_size*all_epsilon/2).pow(2).mean()

                loss.backward()
                optim_score.step()

                logger.log({
                    'score_loss': loss.item(),
                    'ESS': dic['ESS'][-1],
                    'log_z' : dic['log_z'][-1],
                }, step=total_step)
                total_step+=1
            score_function.requires_grad_(False)

            if i == n_i*method_kick_in and i>0:
                print("Switch to snl")
                bias.explicit_bias.data = init_bias(bias, f, sample_q, m, K, 100,).data
            f.train()
                            

            x_p_d = sample_data(m=m)
            x_q, log_weights, dic = sample_q(K, m)  

            log_z = t.logsumexp(bias(f(x_q))-log_weights-math.log(m), dim=0)
            print(log_z)
            print(log_weights)
            print(f(x_q))
            print(dic['ESS_no_score'][-1])
            print(dic['log_z'][-1])
            print(dic['ESS'][-1])
            print(dic['log_z_no_score'][-1])
            # current_ESS = 2*t.logsumexp(target_proba_back - log_weights, dim=0)- t.logsumexp(2*target_proba - 2*log_weights, dim=0)

            ESS = 2*t.logsumexp(f(x_q)-log_weights,dim=0) - t.logsumexp(2*f(x_q)-2*log_weights.detach(), dim=0)
            assert (ESS.item()>0.8), "ESS is not consistent, ESS = {}, ESS_dic = {}".format(ESS.item(), dic["ESS"][-1])
            assert (ESS.item()-dic["ESS"][-1])<1.0, "ESS is not consistent, ESS = {}, ESS_dic = {}".format(ESS.item(), dic["ESS"][-1])
            assert (log_z.item()-dic["log_z"][-1])<1.0, "log_z is not consistent, log_z = {}, log_z_dic = {}".format(log_z.item(), dic["log_z"][-1])
            log_likelihood = bias(f(x_p_d)).mean() - log_z
            snl = bias(f(x_p_d)).mean() - log_z.exp() +1
           

            if i< n_i*method_kick_in:
                loss = -(f(x_p_d).mean() - f(x_q).mean()) # Likelihood f(x)= -E(x)
            else :
                if log_method :
                    loss = -log_likelihood
                else :
                    loss = -snl

            optim.zero_grad()
            (loss).backward(retain_graph=True)
            optim_bias.zero_grad()
            grad_bias = -(log_z.exp() -1)
            bias.explicit_bias.grad = grad_bias.reshape(bias.explicit_bias.shape)
            optim_bias.step()
            optim.step()

            if i%100 == 0:
                f.eval()
                log_z = init_bias(bias, f, sample_q, m, K, 10,)
                total_log_likelihood = 0
                energy_val = 0
                nb_element = 0
                for x_val in iter(dataloader_val):
                    x_val = x_val[0].to(device)
                    batch_size = x_val[0].shape[0]
                    energy_val = (energy_val*nb_element + bias(f(x_val)).mean() * batch_size)/(nb_element+batch_size)
                log_likelihood = energy_val - log_z
                snl = energy_val - log_z.exp() +1
                logger.log({"val/log_likelihood": log_likelihood.item(), "val/snl": snl.item(), "val/energy": energy_val.item(), "val/log_z":log_z.item()}, step=total_step)


            logger.log({'f(x_p_d)': f(x_p_d).mean().item(), 'f(x_q)': f(x_q).mean().item(), 'step': i}, step=total_step)
            logger.log({
                    'grad_norm_max': np.max(dic['f_prime_mean']),
                    'grad_norm_mean': np.mean(dic['f_prime_mean']),
                    'grad_norm_std': np.std(dic['f_prime_mean']),
                    'grad_norm_min':np.min(dic['f_prime_mean']),
                    'bias': bias.explicit_bias.item(),
                    'loss': loss.item(),
                    'log_z': log_z.item(),
                    'log_likelihood': log_likelihood.item(),
                    'snl' : snl.item(),
                    'ESS' : ESS.item(),
                    'log_weights': log_weights.mean().item(),
                }, step=total_step)
            


            if i % 50 == 0 :
                plot_graph(os.path.join(folder, 'normgrad_x_q_{:>06d}.png'.format(i)), dic['f_prime_mean'], dic['f_prime_std'], logger, total_step, name="normgrad")
                plot_graph(os.path.join(folder, 'epsback_{:>06d}.png'.format(i)), dic['eps_back'], dic['eps_forward'], logger, total_step, name="epsback")
                plot_graph(os.path.join(folder, 'epsback_no_score_{:>06d}.png'.format(i)), dic['eps_back_no_score'], dic['eps_forward'], logger, total_step, name="epsback_no_score")
                plot_graph(os.path.join(folder, 'log_acceptance_rate_no_score{:>06d}.png'.format(i)), dic['log_acceptance_rate_no_score'], None, logger, total_step, name="log_acceptance_rate_no_score")
                plot_graph(os.path.join(folder, 'log_acceptance_rate{:>06d}.png'.format(i)), dic['log_acceptance_rate'], None, logger, total_step, name="log_acceptance_rate")

            if i % 50 == 0:
                print('{:>6d} f(x_p_d)={:>14.9f} f(x_q)={:>14.9f}'.format(total_step, f(x_p_d).mean(), f(x_q).mean()))
                plot(os.path.join(folder, 'x_q_{:>06d}.png'.format(i)), x_q, logger, total_step,)
            total_step+=1

        
        wandb.finish()
    except Exception as e:
        logger.log({"error": e})
        wandb.finish()
        raise e
            