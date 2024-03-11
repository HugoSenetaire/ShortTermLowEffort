
import torch.nn as nn

class ConcatEnergyBase(nn.Module):
    def __init__(self, f, base, bias=None):
        super(ConcatEnergyBase, self).__init__()
        self.f = f
        self.base = base
        self.bias = bias

    def forward(self, x):
        energy = self.f(x)
        if self.base is not None:
            energy = energy - self.base.log_prob(x)
        if self.bias is not None:
            energy = energy + self.bias.explicit_bias
        return energy
