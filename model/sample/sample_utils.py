
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


def sample_LangevinDynamicsFullTrajectory(K, m, target_function, step_size, sigma_step, clamp, annealing, device, proposal,):
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
    proposal: function
        function to sample from the base distribution
    """
    f_prime_mean = []
    f_prime_std = []
    eps_back_mean = []
    eps_forward_mean = []
    liste_ESS = []
    log_z = []
    log_acceptance_rate_mean = []
    

    x_k, inv_log_weights, _ = proposal.sample(m)
    x_k = t.autograd.Variable(x_k.to(device), requires_grad=True)
    trajectory = []
    epsilon_trajectory = []
    
    for k in range(1, K):
        # Compute the forward step
        target_proba = target_function(k, K, x_k, )
        f_prime = t.autograd.grad(target_proba.sum(), [x_k], retain_graph=True,  create_graph=False)[0]
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
        target_proba_back = target_function(k, K, x_kp1, )
        f_prime_new = t.autograd.grad(target_proba_back.sum(), [x_kp1], retain_graph=True,  create_graph=False)[0]
        x_back = x_kp1.data - step_size * f_prime_new 
        eps_back = (x_k.data - x_back)/math.sqrt(2*step_size)/sigma_step
        
        
        # Compute the log transition probability
        log_prob_forward = t.distributions.Normal(0, 1).log_prob(epsilon).sum(dim=(1,2,3)).detach()
        log_prob_backward = t.distributions.Normal(0, 1).log_prob(eps_back).sum(dim=(1,2,3)).detach()

        # Compute the acceptance probability
        log_acceptance_rate = -(target_proba+log_prob_forward - log_prob_backward -target_proba_back).logsumexp(0) - math.log(m)

        # Update the log proposal value
        inv_log_weights = inv_log_weights + log_prob_forward - log_prob_backward

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


def sample_LangevinDynamics(K, m, target_function, step_size, sigma_step, clamp, annealing, device, proposal,):
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
    clamp: bool
        whether to clamp the values or not
    annealing: str
        type of annealing process, choices=["annealing", "cosine_annealing", "noannealing"]
    device: str
        device to use
    proposal: function
        function to sample from the base distribution
    """
    f_prime_mean = []
    f_prime_std = []
    eps_back_mean = []
    eps_forward_mean = []
    log_acceptance_rate_mean = []
    log_prob_forward_mean = []
    log_prob_backward_mean = []
    log_z = []
    liste_ESS = []
    
    x_k, inv_log_weights, _ = proposal.sample(m)
    x_k = t.autograd.Variable(x_k.to(device), requires_grad=True)
    for k in range(1, K+1):
        # Compute the forward step
        target_proba = target_function(k, K, x_k, )
        f_prime = t.autograd.grad(target_proba.sum(), [x_k], retain_graph=False,  create_graph=False)[0]
        f_prime_norm = (f_prime.flatten(1) **2).sum(dim=1)
        epsilon = t.randn_like(x_k)
        x_kp1 = x_k.data + step_size * f_prime + math.sqrt(2*step_size) * sigma_step * epsilon

        # x_k.data += f_prime + sigma_step * t.randn_like(x_k)
        if clamp :
            x_kp1 = x_kp1.clamp(0,1)
        x_kp1 = t.autograd.Variable(x_kp1, requires_grad=True)
        
        # Compute the backward step
        target_proba_back = target_function(k, K, x_kp1, )
        f_prime_new = t.autograd.grad(target_proba_back.sum(), [x_kp1], retain_graph=False, create_graph=False,)[0]
        x_back = x_kp1.data - step_size * f_prime_new 
        eps_back = (x_k.data - x_back)/math.sqrt(2*step_size)/sigma_step
        
        
        # Compute the log transition probability
        log_prob_forward = t.distributions.Normal(0, 1).log_prob(epsilon).sum(dim=(1,2,3)).detach()
        log_prob_backward = t.distributions.Normal(0, 1).log_prob(eps_back).sum(dim=(1,2,3)).detach()

        # Compute the acceptance probability
        log_acceptance_rate = -(target_proba+log_prob_forward - log_prob_backward -target_proba_back).logsumexp(0) - math.log(m)




        # Update the log proposal value
        inv_log_weights = inv_log_weights + log_prob_forward - log_prob_backward

        current_log_z = t.logsumexp(target_proba_back - inv_log_weights - math.log(m), dim=0)
        current_ESS = 2*t.logsumexp(target_proba_back - inv_log_weights, dim=0)- t.logsumexp(2*target_proba_back - 2*inv_log_weights, dim=0)

    
        # Logging intermediate values
        eps_back_mean.append(eps_back.flatten(1).pow(2).sum(1).mean().item())
        eps_forward_mean.append(epsilon.flatten(1).pow(2).sum(1).mean().item())
        liste_ESS.append(current_ESS.exp().item())
        log_z.append(current_log_z.item())
        f_prime_mean.append(f_prime_norm.mean().item())
        f_prime_std.append(f_prime_norm.std().item()) 
        log_acceptance_rate_mean.append(log_acceptance_rate.item())
        log_prob_forward_mean.append(log_prob_forward.mean().item())
        log_prob_backward_mean.append(log_prob_backward.mean().item())

        x_k.data = x_kp1.data

    dic = {
            "eps_back": eps_back_mean,
            "eps_forward": eps_forward_mean,
            "log_prob_forward": log_prob_forward_mean,
            "log_prob_backward": log_prob_backward_mean,
            "f_prime_mean": f_prime_mean,
            "f_prime_std": f_prime_std,
            "log_z": log_z,
            "ESS": liste_ESS,
            "log_acceptance_rate": log_acceptance_rate_mean,
            }
    

    return x_k.detach(), inv_log_weights, dic



def sample_LangevinDynamicsScoreControl(K, m, target_function, step_size, sigma_step, clamp, annealing, device, proposal, score_function = None):
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
    proposal: function
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
    score_function_mean = []

    x_k, inv_log_weights, _ = proposal.sample(m)
    x_k = t.autograd.Variable(x_k.to(device), requires_grad=True)
    
    for k in range(1, K):
        # Compute the forward step
        target_proba = target_function(k, K, x_k, )
        f_prime = t.autograd.grad(target_proba.sum(), [x_k], retain_graph=True,  create_graph=False)[0]
        f_prime_norm = (f_prime.flatten(1) **2).sum(dim=1)
        epsilon = t.randn_like(x_k)
        x_kp1 = x_k.data + step_size * f_prime + math.sqrt(2*step_size) * sigma_step * epsilon

        # x_k.data += f_prime + sigma_step * t.randn_like(x_k)
        if clamp == "clamp":
            x_kp1 = x_kp1.clamp(0,1)
        x_kp1 = t.autograd.Variable(x_kp1, requires_grad=True)
        
        # Compute the backward step
        target_proba_back = target_function(k+1, K, x_kp1, )
        f_prime_new = t.autograd.grad(target_proba_back.sum(), [x_kp1], retain_graph=True,  create_graph=False)[0]
        x_back_no_score = x_kp1.data - step_size * f_prime_new 
        eps_back_no_score = (x_k.data - x_back_no_score)/math.sqrt(2*step_size)/sigma_step



        k_tensor = t.full((x_k.shape[0],), k+1).to(device)
        score = score_function(x_kp1, k_tensor)
        x_back = x_kp1.data - step_size * f_prime_new + 2*sigma_step * score.detach()
        score_function_norm = (score.flatten(1) **2).sum(dim=1)
        eps_back = (x_k.data - x_back)/math.sqrt(2*step_size)/sigma_step
        
        
        # Compute the log transition probability
        log_prob_forward = t.distributions.Normal(0, 1).log_prob(epsilon).sum(dim=(1,2,3)).detach()
        log_prob_backward_no_score = t.distributions.Normal(0, 1).log_prob(eps_back_no_score).sum(dim=(1,2,3)).detach()
        log_prob_backward = t.distributions.Normal(0, 1).log_prob(eps_back).sum(dim=(1,2,3)).detach()

        # Compute the acceptance probability
        log_acceptance_rate_no_score = -(target_proba+log_prob_forward - log_prob_backward_no_score -target_proba_back).logsumexp(0) - math.log(m)
        log_acceptance_rate = -(target_proba+log_prob_forward - log_prob_backward -target_proba_back).logsumexp(0) - math.log(m)



        # Update the log proposal value

        inv_log_weights = inv_log_weights + log_prob_forward - log_prob_backward
        current_log_z = t.logsumexp(target_proba_back - inv_log_weights - math.log(m), dim=0)
        current_ESS = 2*t.logsumexp(target_proba_back - inv_log_weights, dim=0)- t.logsumexp(2*target_proba_back - 2*inv_log_weights, dim=0)

        inv_log_weights_no_score = inv_log_weights_no_score + log_prob_forward - log_prob_backward_no_score
        current_log_z_no_score = t.logsumexp(target_proba_back - inv_log_weights_no_score - math.log(m), dim=0)
        current_ESS_no_score = 2*t.logsumexp(target_proba_back - inv_log_weights_no_score, dim=0)- t.logsumexp(2*target_proba_back - 2*inv_log_weights_no_score, dim=0)
    
        # Logging intermediate values forward :
        f_prime_mean.append(f_prime_norm.mean().item())
        f_prime_std.append(f_prime_norm.std().item()) 
        eps_forward_mean.append(epsilon.flatten(1).pow(2).sum(1).mean().item())


        # Logging intermediate values backward and score:
        eps_back_mean.append(eps_back.flatten(1).pow(2).sum(1).mean().item())
        log_z.append(current_log_z.item())
        score_function_mean.append(score_function_norm.mean().item())
        liste_ESS.append(current_ESS.exp().item())
        log_acceptance_rate_mean.append(log_acceptance_rate.item())


        # Logging intermediate values backward no score:
        eps_back_mean_no_score.append(eps_back_no_score.flatten(1).pow(2).sum(1).mean().item())
        log_acceptance_rate_mean_no_score.append(log_acceptance_rate_no_score.item())
        log_z_no_score.append(current_log_z_no_score.item())
        liste_ESS_no_score.append(current_ESS_no_score.exp().item())


        # Update the value of x_k
        x_k.data = x_kp1.data

  

    dic = {
            "eps_back": eps_back_mean,
            "eps_back_no_score": eps_back_mean_no_score,
            "eps_forward": eps_forward_mean,
            "f_prime_mean": f_prime_mean,
            "f_prime_std": f_prime_std,
            "score_function_mean": score_function_mean,
            "log_acceptance_rate": log_acceptance_rate_mean,
            "log_acceptance_rate_no_score": log_acceptance_rate_mean_no_score,
            "log_z": log_z,
            "log_z_no_score": log_z_no_score,
            "ESS": liste_ESS,
            "ESS_no_score": liste_ESS_no_score,
            }
    

    return x_k.detach(), inv_log_weights, dic