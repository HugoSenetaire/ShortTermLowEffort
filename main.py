import torch as t, torch.nn as nn
import torchvision as tv, torchvision.transforms as tr
import tqdm
import matplotlib.pyplot as plt
import math
from torch.nn.utils.parametrizations import spectral_norm
import pathlib
import os
import wandb
import numpy as np



t.cuda.empty_cache()



import argparse

def parser_default():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annealing", default="noannealing", type=str, choices=["annealing", "cosine_annealing", "noannealing"])
    parser.add_argument("--clamp", default="", type=str, choices=["clamp", ""])
    parser.add_argument("--sn", action="store_true", help="sn")
    parser.add_argument("--K", default=100, type=int, help="K")
    parser.add_argument("--n_i", default=501, type=int, help="n_i")
    parser.add_argument("--lambda_reg", default=0.1, type=float, help="lambda_reg")
    parser.add_argument("--n_f", default=64, type=int, help="n_f")
    parser.add_argument("--m", default=8**2, type=int, help="m")
    parser.add_argument("--n_ch", default=1, type=int, help="n_ch")
    parser.add_argument("--im_sz", default=32, type=int, help="im_sz")
    parser.add_argument("--sigma_data", default=3e-2, type=float, help="sigma_data")
    parser.add_argument("--sigma_step", default=1e-2, type=float, help="sigma_step")
    parser.add_argument("--seed", default=1, type=int, help="seed")
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
    seed = args.seed

    t.cuda.empty_cache()
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)


    folder = "./mnist_{}_{}_{}_sn_{}".format(annealing, clamp, K, sn)
    p = pathlib.Path(folder)
    p.mkdir(parents=True, exist_ok=True)
    logger = wandb.init(project="short_term_analysis", config=vars(args), name=folder)

    try :


        device = t.device('cuda' if t.cuda.is_available() else 'cpu')

        class F(nn.Module):
            def __init__(self, n_c=n_ch, n_f=n_f, l=0.2, sn= sn):
                super(F, self).__init__()
                if not sn :
                    self.f = nn.Sequential(
                        nn.Conv2d(n_c, n_f, 3, 1, 1),
                        nn.LeakyReLU(l),
                        nn.Conv2d(n_f, n_f * 2, 4, 2, 1),
                        nn.LeakyReLU(l),
                        nn.Conv2d(n_f * 2, n_f * 4, 4, 2, 1),
                        nn.LeakyReLU(l),
                        nn.Conv2d(n_f * 4, n_f * 8, 4, 2, 1),
                        nn.LeakyReLU(l),
                        nn.Conv2d(n_f * 8, 1, 4, 1, 0))
                else :
                    self.f = nn.Sequential(
                        spectral_norm(nn.Conv2d(n_c, n_f, 3, 1, 1)),
                        nn.LeakyReLU(l),
                        spectral_norm(nn.Conv2d(n_f, n_f * 2, 4, 2, 1)),
                        nn.LeakyReLU(l),
                        spectral_norm(nn.Conv2d(n_f * 2, n_f * 4, 4, 2, 1)),
                        nn.LeakyReLU(l),
                        spectral_norm(nn.Conv2d(n_f * 4, n_f * 8, 4, 2, 1)),
                        nn.LeakyReLU(l),
                        spectral_norm(nn.Conv2d(n_f * 8, 1, 4, 1, 0)))



            def forward(self, x):
                return self.f(x).squeeze()

        f = F(n_c = n_ch, n_f = n_f, sn=sn).to(device)

        def aux_f(k,K, x_k, annealing):
            if annealing == "annealing":
                return f(x_k)*(k+1)/K
            elif annealing == "cosine_annealing":
                return f(x_k)*math.cos((k+1)/K * math.pi*0.5)
            else :
                return f(x_k)







        transform = tr.Compose([tr.Resize(im_sz), tr.ToTensor(),])
        # p_d = t.stack([x[0] for x in tv.datasets.CIFAR10(root='data/cifar10', download=True, transform=transform)]).to(device)
        p_d = t.stack([x[0] for x in tv.datasets.MNIST(root='data/mnist', download=True, transform=transform)]).to(device)

        noise = lambda x: x + sigma_data * t.randn_like(x)
        def sample_p_d():
            p_d_i = t.LongTensor(m).random_(0, p_d.shape[0])
            return noise(p_d[p_d_i]).detach()

        sample_p_0 = lambda: t.FloatTensor(m, n_ch, im_sz, im_sz).uniform_(0, 1).to(device)
        def sample_q(K=K):
            f_prime_mean = []
            f_prime_std = []
            x_k = t.autograd.Variable(sample_p_0(), requires_grad=True)
            for k in range(K):
                f_prime = t.autograd.grad(aux_f(k, K, x_k, annealing).sum(), [x_k], retain_graph=True,  create_graph=False)[0]
                f_prime_norm = (f_prime.flatten(1) **2).sum(dim=1)
                f_prime_mean.append(f_prime_norm.mean().item())
                f_prime_std.append(f_prime_norm.std().item()) 
                x_k.data += f_prime + sigma_step * t.randn_like(x_k)
                if clamp == "clamp":
                    x_k = x_k.clamp(0,1)
            return x_k.detach(), f_prime_mean, f_prime_std

        sqrt = lambda x: int(t.sqrt(t.Tensor([x])))

        def plot(path, x, logger = None, i = 0):
            tv.utils.save_image(t.clamp(x, 0., 1.), path, normalize=True, nrow=sqrt(m))
            if logger :
                logger.log({"x": [wandb.Image(path)]}, step = i)

            
        def plot_graph(path, mean, std,logger= None, i = 0):
            fig = plt.figure()
            plt.plot(mean,)
            plt.xlabel('step k')
            plt.ylabel('gradient_norm')
            plt.savefig(path)
            if logger :
                logger.log({"gradient_norm": [wandb.Image(path)]}, step = i)
            plt.close()

        optim = t.optim.Adam(f.parameters(), lr=1e-4, betas=[.9, .999], )

        for i in tqdm.tqdm(range(n_i)):
            x_p_d = sample_p_d()
            x_q, f_prime_mean, f_prime_std = sample_q()  
            L = f(x_p_d).mean() - f(x_q).mean() 
            # L -= lambda_reg * t.cat(f_prime_mean).sum()
            optim.zero_grad()
            (-L).backward()
            optim.step()

            if i% 10 :
                logger.log({'f(x_p_d)': f(x_p_d).mean().item(), 'f(x_q)': f(x_q).mean().item(), 'step': i}, step=i)
                logger.log({
                        'grad_norm_max': np.max(f_prime_mean) ,
                        'grad_norm_mean': np.mean(f_prime_mean),
                        'grad_norm_std': np.std(f_prime_mean),
                        'grad_norm_min':np.min(f_prime_mean),
                    }, step=i)



            if i % 50 == 0 :
                plot_graph(os.path.join(folder, 'norm_x_q_{:>06d}.png'.format(i)), f_prime_mean, f_prime_std, logger, i)
            if i % 50 == 0:
                print('{:>6d} f(x_p_d)={:>14.9f} f(x_q)={:>14.9f}'.format(i, f(x_p_d).mean(), f(x_q).mean()))
                plot(os.path.join(folder, 'x_q_{:>06d}.png'.format(i)), x_q, logger, i)
        wandb.finish()
    except Exception as e:
        logger.log({"error": e})
        wandb.finish()
        raise e
            