import math        

def target_function(k,K, x_k, annealing, energy, proposal=None, use_proposal=False):
    """
    Utility function to compute the target function for the annealing process

    Args:
    -----
    k (int): current step
    K (int): total number of steps
    x_k (torch.Tensor): current state
    annealing (str): type of annealing process, choices=["annealing", "cosine_annealing", "noannealing"]
    f (function): function to compute the energy

    Returns:
    --------
    float: value of the target function
    """
    assert k>0 and k<=K, "The current step must be between 1 and K"
    if annealing == "annealing":
        if use_proposal:
            assert proposal is not None, "The proposal must be provided"
            return -energy(x_k)*k/K + proposal.log_prob(x_k)*(1-k/K)
        else:
            return -energy(x_k)*k/K 
    elif annealing == "cosine_annealing":
        if use_proposal:
            assert proposal is not None, "The proposal must be provided"
            return -energy(x_k)*math.sin(k/K * math.pi*0.5) + proposal.log_prob(x_k)*(1-math.sin(k/K * math.pi*0.5))
        else:
            return -energy(x_k)*math.sin(k/K * math.pi*0.5)
    else :
        if use_proposal :
            raise ValueError("The annealing process does not allow the use of a proposal")
        return -energy(x_k)