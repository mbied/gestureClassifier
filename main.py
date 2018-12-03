import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pydmps.dmp_discrete

def calc_similarity(w1, w2):
    return np.dot(np.transpose(w1),w2)/(np.linalg.norm(w1)*np.linalg.norm(w2))

def get_letter(drawings, alphabet_index, letter_index, set_index):
    # drawings = D.get('drawings')
    alphabet = drawings[alphabet_index][0]
    one_letter_set = alphabet[letter_index][0]
    letter = one_letter_set[set_index][0]
    data = np.transpose(letter[0][0])
    return data



def error_plot(ax, title, x_data, y_data, x_name, y_name):
    ax.plot(x_data, y_data)
    # do other stuff to the axes



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
    dic_weights = {}

    for letter_index in range(number_used_characters):
        dic_weights[letter_index] = {}
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
            w = np.reshape(dmp.w, (60,1))
            dic_weights[letter_index][set_index] = w

        #fig.tight_layout()
        #fig.canvas.flush_events()
        #time.sleep(0.01)
        #if i == 25:
        #    fig.set_size_inches(3, 3)
    #plt.ioff()



    #fig2, ax2 = plt.subplots()
    #ax2.plot(data[0], data[1])
    #ax2.set_title('DMP system - draw loaded character')
    #ax2.plot(y_track[:, 0], y_track[:, 1], 'r--', lw=1)
    #ax2.legend(['original trajectory', 'reproduced trajectory'])

    fig3, ax3 = plt.subplots()
    #fig, _axs = plt.subplots(nrows=2, ncols=2)
    #axs = _axs.flatten()

    np.random.seed(765955)
    similarity_mat_dim = number_used_characters*used_set_size
    mat_similarity = np.empty((similarity_mat_dim, similarity_mat_dim))

    for letter_index_row in range(number_used_characters):
        for set_index_row in range(used_set_size):
            row = letter_index_row*used_set_size+ set_index_row
            for letter_index_col in range(number_used_characters):
                for set_index_col in range(used_set_size):
                    col = letter_index_col*used_set_size + set_index_col
                    w1 = dic_weights[letter_index_row][set_index_row]
                    w2 = dic_weights[letter_index_col][set_index_col]
                    sim = calc_similarity(w1, w2)
                    mat_similarity[row, col] = sim

    ax3 = plt.imshow(mat_similarity, cmap='autumn', interpolation='nearest')


    plt.colorbar(ax3)



    plt.show()
    print('sucess')

    #plt.plot(data[0], data[1])
    #plt.show()

    #fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)

    #error_plot(ax1, "Fig_1", [1, 2, 3], [[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    #error_plot(ax2, "Fig_2", [a, b, c], [[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    #error_plot(ax3, "Fig_3", [2a, 2b, 3c], [[1, 1, 1], [2, 2, 2], [3, 3, 3]])