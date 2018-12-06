from scipy.stats import multivariate_normal
import numpy as np

def calc_responsibility(w, mu_dict, sigma_dict):
    K = len(mu_dict)
    pi_k = 1 / K
    r = np.empty(K)
    for k in range(K):
        mu = mu_dict[k]
        sigma = sigma_dict[k]
        rv = multivariate_normal(mu, sigma, allow_singular=True)
        r_k = rv.pdf(w)
        r[k] = r_k

    r = r/np.sum(r)

    return r