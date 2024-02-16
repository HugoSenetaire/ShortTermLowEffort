
import wandb
import torch as t
import torchvision as tv
import matplotlib.pyplot as plt
import math
import os
sqrt = lambda x: int(t.sqrt(t.Tensor([x])))

def plot(path, x, logger = None, i = 0, name="x"):
            tv.utils.save_image(t.clamp(x, 0., 1.), path, normalize=True, nrow=sqrt(x.shape[0]))
            if logger :
                logger.log({name: [wandb.Image(path)]}, step = i)

            
def plot_graph(path, mean, std,logger= None, i = 0, name = "norm"):
    fig = plt.figure()
    plt.plot(mean,)
    plt.xlabel('step k')
    plt.ylabel(name)
    plt.savefig(path)
    logger.log({name: [wandb.Image(path)]}, step = i)
    plt.close()