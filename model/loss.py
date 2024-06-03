import torch
import numpy as np


def nll_loss(pred_mu, pred_cov, y, return_numpy=True):
    pred_std = torch.sqrt(pred_cov)
    gaussian = torch.distributions.Normal(pred_mu, pred_std)
    nll = torch.mean(-gaussian.log_prob(y))
    if return_numpy:
        return nll.cpu().detach().numpy()
    return nll

def rmse_metric(y_pred, y_true):
    """
    Calculate the metric by specific axis
    """
    return np.sqrt(np.mean(np.square(y_pred - y_true)))

def kld_gaussian_loss(z_mean_all, z_var_all, z_mean_context, z_var_context):
    std_all = torch.sqrt(z_var_all)
    std_context = torch.sqrt(z_var_context)

    dist_all = torch.distributions.Normal(z_mean_all, std_all)
    dist_context = torch.distributions.Normal(z_mean_context, std_context)

    kld = torch.distributions.kl_divergence(dist_all, dist_context)
    return torch.mean(kld)

