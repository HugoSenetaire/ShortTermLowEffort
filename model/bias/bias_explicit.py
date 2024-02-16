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
            return x + self.explicit_bias
        



def init_bias(bias, f, sample_q, batch_size, K, nb_batch,):
    """
    Initialize the bias of the energy function using nb_sample samples from the proposal.

    Args:
    """

    total_log_z = t.tensor(0, dtype = t.float32, device = bias.explicit_bias.device, requires_grad=False)
    f.eval()
    nb_batch=5
    nb_sample = batch_size * nb_batch
    for k in tqdm.tqdm(range(nb_batch)):

        x_q, log_prob, dic = sample_q(K, batch_size)
        current_log_z = t.logsumexp(f(x_q) - log_prob, dim=0)
        total_log_z = t.logsumexp(t.stack([total_log_z, current_log_z], dim=0), dim=0)




    total_log_z = total_log_z - math.log(nb_sample)

    return total_log_z
