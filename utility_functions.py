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

def calc_similarity(w1, w2):
    return np.dot(np.transpose(w1),w2)/(np.linalg.norm(w1)*np.linalg.norm(w2))

def get_letter(drawings, alphabet_index, letter_index, set_index):
    # drawings = D.get('drawings')
    alphabet = drawings[alphabet_index][0]
    one_letter_set = alphabet[letter_index][0]
    letter = one_letter_set[set_index][0]
    data = np.transpose(letter[0][0])
    return data

def get_mat_from_dict(dict):
    D = len(dict)
    n = len(dict[0])
    M = np.zeros((D,n))
    for i in range(D):
        M[i] = dict[i].flatten()

    return M


def error_plot(ax, title, x_data, y_data, x_name, y_name):
    ax.plot(x_data, y_data)
    # do other stuff to the axes