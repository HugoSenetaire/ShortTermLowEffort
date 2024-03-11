from .abstract_proposal import AbstractProposal
import torch as t
import torch.nn as nn

class UniformProposal(AbstractProposal):
    def __init__(self, dataset,):
        super(UniformProposal, self).__init__()
        low = t.min(dataset, dim=0)
        full_zero = t.zeros_like(low.values)
        low = t.where(low.values<0, low.values, full_zero)
        high = t.max(dataset, dim=0)
        full_one = t.ones_like(high.values)
        high = t.where(high.values>1, high.values, full_one)
        print(low, high)
        self.low = nn.Parameter(low, requires_grad=False)
        self.high = nn.Parameter(high, requires_grad=False)
        assert self.low.shape==self.high.shape, "The shape of the low and the high are not consistent"

    def sample(self, batch_size,):
        x = t.rand(batch_size, *self.low.shape, device=self.low.device)*(self.high-self.low) + self.low
        return x, self.log_prob(x), {}
    
    def log_prob(self, x):
        assert x.shape[1:]==self.low.shape, "The shape of the samples is not consistent with the shape of the proposal"
        assert x.device==self.low.device, "The device of the samples is not consistent with the device of the proposal"
        return (self.high-self.low).log().sum().reshape(1).expand(x.shape[0])