
import torch as t  
import math


def noise(x, sigma_data):
    return x + sigma_data * t.randn_like(x)


def sample_p_d(m, p_d, noise_function=lambda x: noise(x, sigma_data=0.1),):
    """
    Sample from the data distribution
    """
    p_d_i = t.LongTensor(m).random_(0, p_d.shape[0])
    return noise_function(p_d[p_d_i], ).detach()

def sample_base(m, n_ch, im_sz):
    """
    Sample from the base distribution here a uniform only

    Args:
    -----
    m: int
        batch size
    n_ch: int
        number of channels
    im_sz: int  
        size of the image
    """
    return t.FloatTensor(m, n_ch, im_sz, im_sz).uniform_(0, 1)

def sample_LangevinDynamicsFullTrajectory(K, m, target_function, step_size, sigma_step, clamp, annealing, device, sample_base=sample_base,):
    """
    Sample from the distribution q using the Langevin dynamics

    Args:
    -----
    K: int
        number of steps
    m: int
        batch size
    target_function: function
        function to compute the target function
    step_size: float
        step size for the Langevin dynamics
    sigma_step: float
        standard deviation for the Langevin dynamics
    clamp: str
        whether to clamp the values or not
    annealing: str
        type of annealing process, choices=["annealing", "cosine_annealing", "noannealing"]
    device: str
        device to use
    sample_base: function
        function to sample from the base distribution
    """
    f_prime_mean = []
    f_prime_std = []
    eps_back_mean = []
    eps_forward_mean = []
    liste_ESS = []
    log_z = []
    log_acceptance_rate_mean = []
    x_k = t.autograd.Variable(sample_base(m).to(device), requires_grad=True)
    log_weights = t.zeros(m, device=device, requires_grad=False)
    trajectory = []
    epsilon_trajectory = []
    
    for k in range(K):
        # Compute the forward step
        target_proba = target_function(k, K, x_k, )
        f_prime = t.autograd.grad(target_function(k, K, x_k,).sum(), [x_k], retain_graph=True,  create_graph=False)[0]
        f_prime_norm = (f_prime.flatten(1) **2).sum(dim=1)
        epsilon = t.randn_like(x_k)
        x_kp1 = x_k.data + step_size * f_prime + math.sqrt(2*step_size) * sigma_step * epsilon

        # x_k.data += f_prime + sigma_step * t.randn_like(x_k)
        if clamp == "clamp":
            x_kp1 = x_kp1.clamp(0,1)
        epsilon_trajectory.append(epsilon)
        trajectory.append(x_kp1)
        x_kp1 = t.autograd.Variable(x_kp1, requires_grad=True)
        
        # Compute the backward step
        f_prime_new = t.autograd.grad(target_function(k, K, x_kp1, ).sum(), [x_kp1], retain_graph=True,  create_graph=False)[0]
        x_back = x_kp1.data - step_size * f_prime_new 
        eps_back = (x_k.data - x_back)/math.sqrt(2*step_size)/sigma_step
        target_proba_back = target_function(k+1, K, x_kp1, )
        
        
        # Compute the log transition probability
        log_prob_forward = t.distributions.Normal(0, 1).log_prob(epsilon).sum(dim=(1,2,3)).detach()
        log_prob_backward = t.distributions.Normal(0, 1).log_prob(eps_back).sum(dim=(1,2,3)).detach()

        # Compute the acceptance probability
        log_acceptance_rate = -(target_proba+log_prob_forward - log_prob_backward -target_proba_back).logsumexp(0) - math.log(m)




        # Update the log proposal value
        log_weights = log_weights + log_prob_forward - log_prob_backward

        current_log_z = t.logsumexp(target_proba_back - log_prob_forward - math.log(m), dim=0)
        current_ESS = 2*t.logsumexp(target_proba_back - log_prob_forward, dim=0)- t.logsumexp(2*target_proba_back - 2*log_prob_forward, dim=0)

    
        # Logging intermediate values
        eps_back_mean.append(eps_back.flatten(1).pow(2).sum(1).mean().item())
        eps_forward_mean.append(epsilon.flatten(1).pow(2).sum(1).mean().item())
        liste_ESS.append(current_ESS.exp().item())
        log_z.append(current_log_z.item())
        f_prime_mean.append(f_prime_norm.mean().item())
        f_prime_std.append(f_prime_norm.std().item()) 
        log_acceptance_rate_mean.append(log_acceptance_rate.item())
        x_k.data = x_kp1.data

    dic = {
            "eps_back": eps_back_mean,
            "eps_forward": eps_forward_mean,
            "f_prime_mean": f_prime_mean,
            "f_prime_std": f_prime_std,
            "log_z": log_z,
            "ESS": liste_ESS,
            "log_acceptance_rate": log_acceptance_rate_mean,
            }


    return trajectory, epsilon_trajectory, dic


def sample_LangevinDynamics(K, m, target_function, step_size, sigma_step, clamp, annealing, device, sample_base=sample_base,):
    """
    Sample from the distribution q using the Langevin dynamics

    Args:
    -----
    K: int
        number of steps
    m: int
        batch size
    target_function: function
        function to compute the target function
    step_size: float
        step size for the Langevin dynamics
    sigma_step: float
        standard deviation for the Langevin dynamics
    clamp: str
        whether to clamp the values or not
    annealing: str
        type of annealing process, choices=["annealing", "cosine_annealing", "noannealing"]
    device: str
        device to use
    sample_base: function
        function to sample from the base distribution
    """
    f_prime_mean = []
    f_prime_std = []
    eps_back_mean = []
    eps_forward_mean = []
    log_acceptance_rate_mean = []
    log_z = []
    liste_ESS = []
    x_k = t.autograd.Variable(sample_base(m).to(device), requires_grad=True)
    log_weights = t.zeros(m, device=device, requires_grad=False)
    
    for k in range(K):

        # Compute the forward step
        target_proba = target_function(k, K, x_k, )
        f_prime = t.autograd.grad(target_function(k, K, x_k,).sum(), [x_k], retain_graph=True,  create_graph=False)[0]
        f_prime_norm = (f_prime.flatten(1) **2).sum(dim=1)
        epsilon = t.randn_like(x_k)
        x_kp1 = x_k.data + step_size * f_prime + math.sqrt(2*step_size) * sigma_step * epsilon

        # x_k.data += f_prime + sigma_step * t.randn_like(x_k)
        if clamp == "clamp":
            x_kp1 = x_kp1.clamp(0,1)
        x_kp1 = t.autograd.Variable(x_kp1, requires_grad=True)
        
        # Compute the backward step
        f_prime_new = t.autograd.grad(target_function(k, K, x_kp1, ).sum(), [x_kp1], retain_graph=True,  create_graph=False)[0]
        x_back = x_kp1.data - step_size * f_prime_new 
        eps_back = (x_k.data - x_back)/math.sqrt(2*step_size)/sigma_step
        target_proba_back = target_function(k+1, K, x_kp1, )
        
        
        # Compute the log transition probability
        log_prob_forward = t.distributions.Normal(0, 1).log_prob(epsilon).sum(dim=(1,2,3)).detach()
        log_prob_backward = t.distributions.Normal(0, 1).log_prob(eps_back).sum(dim=(1,2,3)).detach()

        # Compute the acceptance probability
        log_acceptance_rate = -(target_proba+log_prob_forward - log_prob_backward -target_proba_back).logsumexp(0) - math.log(m)




        # Update the log proposal value
        log_weights = log_weights + log_prob_forward - log_prob_backward

        current_log_z = t.logsumexp(target_proba_back - log_weights - math.log(m), dim=0)
        current_ESS = 2*t.logsumexp(target_proba_back - log_weights, dim=0)- t.logsumexp(2*target_proba_back - 2*log_weights, dim=0)

    
        # Logging intermediate values
        eps_back_mean.append(eps_back.flatten(1).pow(2).sum(1).mean().item())
        eps_forward_mean.append(epsilon.flatten(1).pow(2).sum(1).mean().item())
        liste_ESS.append(current_ESS.exp().item())
        log_z.append(current_log_z.item())
        f_prime_mean.append(f_prime_norm.mean().item())
        f_prime_std.append(f_prime_norm.std().item()) 
        log_acceptance_rate_mean.append(log_acceptance_rate.item())
        x_k.data = x_kp1.data

    dic = {
            "eps_back": eps_back_mean,
            "eps_forward": eps_forward_mean,
            "f_prime_mean": f_prime_mean,
            "f_prime_std": f_prime_std,
            "log_z": log_z,
            "ESS": liste_ESS,
            "log_acceptance_rate": log_acceptance_rate_mean,
            }
    

    return x_k.detach(), log_weights, dic



def sample_LangevinDynamicsScoreControl(K, m, target_function, step_size, sigma_step, clamp, annealing, device, sample_base=sample_base, score_function = None):
    """
    Sample from the distribution q using the Langevin dynamics

    Args:
    -----
    K: int
        number of steps
    m: int
        batch size
    target_function: function
        function to compute the target function
    step_size: float
        step size for the Langevin dynamics
    sigma_step: float
        standard deviation for the Langevin dynamics
    clamp: str
        whether to clamp the values or not
    annealing: str
        type of annealing process, choices=["annealing", "cosine_annealing", "noannealing"]
    device: str
        device to use
    sample_base: function
        function to sample from the base distribution
    """
    f_prime_mean = []
    f_prime_std = []
    eps_back_mean = []
    eps_forward_mean = []
    eps_back_mean_no_score = []
    log_acceptance_rate_mean = []
    log_acceptance_rate_mean_no_score = []
    log_z = []
    liste_ESS = []
    log_z_no_score = []
    liste_ESS_no_score = []

    x_k = t.autograd.Variable(sample_base(m).to(device), requires_grad=True)
    log_weights = t.zeros(m, device=device, requires_grad=False)
    
    for k in range(K):

        # Compute the forward step
        f_prime = t.autograd.grad(target_function(k, K, x_k,).sum(), [x_k], retain_graph=True,  create_graph=False)[0]
        f_prime_norm = (f_prime.flatten(1) **2).sum(dim=1)
        epsilon = t.randn_like(x_k)
        x_kp1 = x_k.data + step_size * f_prime + math.sqrt(2*step_size) * sigma_step * epsilon

        # x_k.data += f_prime + sigma_step * t.randn_like(x_k)
        if clamp == "clamp":
            x_kp1 = x_kp1.clamp(0,1)
        x_kp1 = t.autograd.Variable(x_kp1, requires_grad=True)
        
        # Compute the backward step
        target_proba = target_function(k, K, x_kp1, )
        f_prime_new = t.autograd.grad(target_proba.sum(), [x_kp1], retain_graph=True,  create_graph=False)[0]
        k_tensor = t.full((x_k.shape[0],), k+1).to(device)
        x_back = x_kp1.data - step_size * f_prime_new - 2*sigma_step * score_function(x_kp1, k_tensor).detach()
        x_back_no_score = x_kp1.data - step_size * f_prime_new
        eps_back = (x_k.data - x_back)/math.sqrt(2*step_size)/sigma_step
        eps_back_no_score = (x_k.data - x_back_no_score)/math.sqrt(2*step_size)/sigma_step
        
        # Compute the log transition probability
        log_prob_forward = t.distributions.Normal(0, 1).log_prob(epsilon).sum(dim=(1,2,3)).detach()
        log_prob_backward = t.distributions.Normal(0, 1).log_prob(eps_back).sum(dim=(1,2,3)).detach()
        log_prob_backward_no_score = t.distributions.Normal(0, 1).log_prob(eps_back_no_score).sum(dim=(1,2,3)).detach()

        # Compute the acceptance probability
        target_proba_back = target_function(k+1, K, x_kp1, )
        log_acceptance_rate = -(target_proba + log_prob_forward - target_proba_back - log_prob_backward).logsumexp(0) - math.log(m)
        log_acceptance_rate_no_score = -(target_proba + log_prob_forward - target_proba_back - log_prob_backward_no_score).logsumexp(0) - math.log(m)



        # Update the log proposal value
        log_weights = log_weights + log_prob_forward - log_prob_backward
        log_weights_no_score = log_weights + log_prob_forward - log_prob_backward_no_score

        # Compute the log partition function
        current_log_z = t.logsumexp(target_proba_back - log_weights - math.log(m), dim=0)
        current_log_z_no_score = t.logsumexp(target_proba_back -log_weights_no_score - math.log(m), dim=0)
        current_ESS = 2*t.logsumexp(target_proba_back - log_weights, dim=0)- t.logsumexp(2*target_proba - 2*log_weights, dim=0)
        current_ESS_no_score = 2*t.logsumexp(target_proba_back - log_weights_no_score, dim=0)- t.logsumexp(2*target_proba - 2*log_weights_no_score, dim=0)


    
        # Logging intermediate values
        eps_back_mean.append(eps_back.flatten(1).pow(2).sum(1).mean().item())
        eps_forward_mean.append(epsilon.flatten(1).pow(2).sum(1).mean().item())
        eps_back_mean_no_score.append(eps_back_no_score.flatten(1).pow(2).sum(1).mean().item())
        f_prime_mean.append(f_prime_norm.mean().item())
        f_prime_std.append(f_prime_norm.std().item()) 
        log_acceptance_rate_mean.append(log_acceptance_rate.item())
        log_acceptance_rate_mean_no_score.append(log_acceptance_rate_no_score.item())
        log_z.append(current_log_z.item())
        liste_ESS.append(current_ESS.exp().item())
        log_z_no_score.append(current_log_z_no_score.item())
        liste_ESS_no_score.append(current_ESS_no_score.exp().item())

        x_k.data = x_kp1.data

    dic = {
            "eps_back": eps_back_mean,
            "eps_back_no_score": eps_back_mean_no_score,
            "eps_forward": eps_forward_mean,
            "f_prime_mean": f_prime_mean,
            "f_prime_std": f_prime_std,
            "log_acceptance_rate": log_acceptance_rate_mean,
            "log_acceptance_rate_no_score": log_acceptance_rate_mean_no_score,
            "log_z": log_z,
            "ESS": liste_ESS,
            "log_z_no_score": log_z_no_score,
            "ESS_no_score": liste_ESS_no_score,
            }
    

    return x_k.detach(), log_weights, dic