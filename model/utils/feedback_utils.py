

from backpack import backpack, extend
from backpack.extensions import (
    GGNMP,
    HMP,
    KFAC,
    KFLR,
    KFRA,
    PCHMP,
    BatchDiagGGNExact,
    BatchDiagGGNMC,
    BatchDiagHessian,
    BatchGrad,
    BatchL2Grad,
    DiagGGNExact,
    DiagGGNMC,
    DiagHessian,
    SqrtGGNExact,
    SqrtGGNMC,
    SumGradSquared,
    Variance,
)
import torch


def feedback_loss(fnn, optim, loss, name):
    dic = {}
    optim.zero_grad()
    with backpack(Variance(), BatchL2Grad()):
        loss.backward(retain_graph=True)
    # for param in fnn.parameters():
        # print(param)
        # break
    variance = torch.stack([param.variance.flatten() for param in fnn.parameters()])
    batch_l2_grad = torch.stach([param.batch_l2_grad.flatten() for param in fnn.parameters()])
    dic[f"var_train/{name}_variance_mean"] = torch.mean(torch.stack(variance))
    dic[f"var_train/{name}_variance_std"] = torch.std(torch.stack(variance))
    dic[f"var_train/{name}_variance_max"] = torch.max(torch.stack(variance))
    dic[f"var_train/{name}_variance_min"] = torch.min(torch.stack(variance))

    dic[f"l2_train/{name}_batch_l2_grad_mean"] = torch.mean(torch.stack(batch_l2_grad))
    dic[f"l2_train/{name}_batch_l2_grad_std"] = torch.std(torch.stack(batch_l2_grad))
    dic[f"l2_train/{name}_batch_l2_grad_max"] = torch.max(torch.stack(batch_l2_grad))
    dic[f"l2_train/{name}_batch_l2_grad_min"] = torch.min(torch.stack(batch_l2_grad))
    return dic