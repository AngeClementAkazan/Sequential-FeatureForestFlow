"This Script compute the flow matching for given data x and time t"
import numpy as np
import os
import copy
import torch

def resh_t(t, x):
    if isinstance(t, float):
        return t
    return t.reshape(-1, *([1] * (len(x.shape) - 1)))

class CFM:
    def __init__(self, sigma: float = 0.0):
        self.sigma = sigma
        # self.t=t
    def mu_t(self, x0, x1, t):
        t = resh_t(t, x0)
        return t * x1 + (1 - t) * x0

    def compute_sigma_t(self, t):
        del t
        return self.sigma

    def sample_xt(self, x0, x1, t, epsilon):
        mu_t = self.mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = resh_t(sigma_t, x0)
        return mu_t + sigma_t * epsilon

    def compute_conditional_flow(self, x0, x1, t, xt):
        del t, xt
        return x1 - x0

    def sample_noise_like(self, x):
        return np.random.randn(*x.shape)

    def Extr_CFM(self, x0, x1,t, return_noise=False):
        # t = torch.rand(x0.shape[0]).type_as(x0)
        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut

    def compute_lambda(self, t):
        sigma_t = self.compute_sigma_t(t)
        return 2 * sigma_t / (self.sigma**2 + 1e-8)