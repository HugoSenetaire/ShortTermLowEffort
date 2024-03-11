from abc import ABC, abstractmethod
import torch.nn as nn

class AbstractProposal(ABC, nn.Module):
    """
    Abstract class for the proposal
    """
    def __init__(self):
        super(AbstractProposal, self).__init__()

    @abstractmethod
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
        pass

    @abstractmethod
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

        pass


