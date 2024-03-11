import torch.nn as nn
import torch as t
import tqdm 
import math
import time
class BiasExplicit(nn.Module):
        def __init__(self):
            super(BiasExplicit, self).__init__()
            self.explicit_bias = nn.Parameter(t.zeros(1, dtype = t.float32), requires_grad=True)
        def forward(self, x):
            return x - self.explicit_bias
        



def init_bias(energy, sample_q, batch_size, K, nb_batch,):
    """
    Initialize the bias of the energy function using nb_sample samples from the proposal.

    Args:
    """

    

    energy.eval()

    nb_sample = batch_size * nb_batch

    x_q, inv_log_weights, dic = sample_q(K, batch_size)  

    total_log_z = t.logsumexp(-energy(x_q) - inv_log_weights, dim=0).detach()
    for k in tqdm.tqdm(range(1,nb_batch)):
        x_q, inv_log_weights, dic = sample_q(K, batch_size)
        current_log_z = t.logsumexp(-energy(x_q) - inv_log_weights, dim=0).detach()
        total_log_z = t.logsumexp(t.stack([total_log_z.reshape(1), current_log_z.reshape(1)], dim=0), dim=0)
    total_log_z = total_log_z - math.log(nb_sample)

    return total_log_z
