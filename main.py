from utility_functions import calc_responsibility, calc_similarity, get_letter, get_mat_from_dict, error_plot

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pydmps.dmp_discrete
import json

if __name__ == "__main__":
    D = sio.loadmat('data/data_background.mat')
    drawings = D.get('drawings')
    alphabet_index = 13
    letter_index = 0
    set_index = 0
    data = get_letter(drawings, alphabet_index, letter_index, set_index)
    number_used_characters = 9
    nrow = np.ceil(np.sqrt(number_used_characters))
    used_set_size = 5
    #ncol = nrow

    fig = plt.figure()
    fig2 = plt.figure()

    weights_per_dimension = 30
    weights_dict = {}

    for letter_index in range(number_used_characters):
        weights_dict[letter_index] = {}
        ax = fig.add_subplot(3, 3, letter_index + 1)
        ax2 = fig2.add_subplot(3, 3, letter_index + 1)

        for set_index in range(used_set_size):
            data = get_letter(drawings, alphabet_index, letter_index, set_index)
            ax.plot(data[0], data[1])


            y_des = data
            # test normal run
            dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=weights_per_dimension, ay=np.ones(2) * 10.0)
            y_track = []
            dy_track = []
            ddy_track = []

            dmp.imitate_path(y_des=y_des, plot=False)
            y_track, dy_track, ddy_track = dmp.rollout()
            ax2.plot(y_track[:, 0], y_track[:, 1],linestyle= '--', lw=1)

            #ax2.legend(['original trajectory', 'reproduced trajectory'])
            w = np.reshape(dmp.w, (60))
            weights_dict[letter_index][set_index] = w   # storing in dict instead of mat to be able
                                                       # to change to unequal sizes of demonstration sets
    sigma_dict = {}
    mu_dict = {}
    eigenvalues_dict = {}
    dtype = np.float64
    for letter_index in range(number_used_characters):
        A = weights_dict[letter_index]
        M = get_mat_from_dict(A)
        C = np.cov(np.transpose(M))
        min_eig = np.min(np.real(np.linalg.eigvals(C)))

        # ensure matrix is positive semidefinite
        if min_eig < 0:
            C -= 10*min_eig * np.eye(*C.shape)



        mu = np.mean(M, axis=0)
        sigma_dict[letter_index] = C
        mu_dict[letter_index] = mu
        eigenvalues_dict[letter_index] = min_eig

    fig3, ax3 = plt.subplots()

    np.random.seed(765955)
    similarity_mat_dim = number_used_characters*used_set_size
    mat_similarity = np.empty((similarity_mat_dim, similarity_mat_dim))

    for letter_index_row in range(number_used_characters):
        for set_index_row in range(used_set_size):
            row = letter_index_row*used_set_size+ set_index_row
            for letter_index_col in range(number_used_characters):
                for set_index_col in range(used_set_size):
                    col = letter_index_col*used_set_size + set_index_col
                    w1 = weights_dict[letter_index_row][set_index_row]
                    w2 = weights_dict[letter_index_col][set_index_col]
                    sim = calc_similarity(w1, w2)
                    mat_similarity[row, col] = sim

    ax3 = plt.imshow(mat_similarity, cmap='autumn', interpolation='nearest')


    plt.colorbar(ax3)

    r = calc_responsibility(w, mu_dict, sigma_dict)

    np.save('./data/sigma_dict.npy', sigma_dict)
    np.save('./data/mu_dict.npy', mu_dict)

    fig4 = plt.figure()
    ax4 = fig4.add_subplot(1, 1, 1)
    ax4.bar(range(9), r, yerr=0.005)
    ax4.set_xticklabels(['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 'iota'])
    ax4.set_ylabel('probability of character')

    plt.show()
    print('success')