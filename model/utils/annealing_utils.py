import math        

def target_function(k,K, x_k, annealing, f):
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
    if annealing == "annealing":
        return f(x_k)*(k+1)/K
    elif annealing == "cosine_annealing":
        return f(x_k)*math.cos((k+1)/K * math.pi*0.5)
    else :
        return f(x_k)