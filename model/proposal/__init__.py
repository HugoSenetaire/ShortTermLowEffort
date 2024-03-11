from .gaussian_proposal import GaussianProposal
from .uniform_proposal import UniformProposal

def getter_proposal(proposal_name, dataset):
    """
    Getter for the proposal

    Args:
    -----
    proposal_name (str): name of the proposal

    Returns:
    --------
    AbstractProposal: proposal
    """
    if proposal_name is None :
        return None
    elif proposal_name=="gaussian":
        return GaussianProposal(dataset)
    elif proposal_name=="uniform":
        return UniformProposal(dataset)
    else:
        raise ValueError("The proposal name is not valid")
    

