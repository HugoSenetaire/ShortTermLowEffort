
from .abstract_proposal import AbstractProposal
import torch as t
import torch.nn as nn

class GaussianProposal(AbstractProposal):
    """
    Class for the Gaussian proposal
    """
    def __init__(self, dataset,):
        """
        Constructor for the Gaussian proposal

        Args:
        -----
        mu (torch.Tensor): mean of the Gaussian
        sigma (torch.Tensor): standard deviation of the Gaussian
        """
        super(GaussianProposal, self).__init__()

        self.mu = t.mean(dataset, dim=0)
        self.sigma = t.std(dataset, dim=0)
        self.mu = nn.Parameter(self.mu, requires_grad=False)
        self.sigma = nn.Parameter(self.sigma, requires_grad=False)
        assert self.mu.shape==self.sigma.shape, "The shape of the mean and the standard deviation are not consistent"

    def sample(self, batch_size,):
        """
        Sample from the proposal

        Args:
        -----
        batch_size (int): size of the batch

        Returns:
        --------
        torch.Tensor: samples
        torch.Tensor: log probability of the samples
        dict: dictionary of the samples
        """
        x = self.mu + self.sigma * t.randn(batch_size, *self.mu.shape, device=self.mu.device)
        x = t.distributions.Normal(self.mu, self.sigma).sample((batch_size,))
        return x, self.log_prob(x), {}

    def log_prob(self, x):
        """
        Compute the log probability of the samples

        Args:
        -----
        x (torch.Tensor): samples

        Returns:
        --------
        torch.Tensor: log probability
        """
        assert x.shape[1:]==self.mu.shape, "The shape of the samples is not consistent with the shape of the proposal"
        assert x.device==self.mu.device, "The device of the samples is not consistent with the device of the proposal"
        return t.distributions.Normal(self.mu, self.sigma).log_prob(x).flatten(1).sum(dim=1, keepdim=False)