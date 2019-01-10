import scipy.io as sio
from utility_functions import calc_responsibility, calc_similarity, get_letter, get_mat_from_dict, error_plot
import numpy as np
import pydmps.dmp_discrete
import matplotlib.pyplot as plt

D = sio.loadmat('data/data_background.mat')
drawings = D.get('drawings')
alphabet_index = 13
letter_index = 0
set_index = 0
data = get_letter(drawings, alphabet_index, letter_index, set_index)
number_used_characters = 9
nrow = np.ceil(np.sqrt(number_used_characters))
used_set_size = 5
ncol = nrow

fig = plt.figure()
fig2 = plt.figure()

weights_per_dimension: int = 50
weights_dict = {}

bad_letter = get_letter(drawings, alphabet_index, letter_index, 3) # only a half circle
alpha_letter = get_letter(drawings, alphabet_index, letter_index, 4) # good alpha

bad_letter = get_letter(drawings, alphabet_index, 7, 3) # only a half circle
alpha_letter = get_letter(drawings, alphabet_index, letter_index, 4) # good alpha

ax = fig.add_subplot(1, 1, 1)
ax.plot(alpha_letter[0], alpha_letter[1])
ax.plot(bad_letter[0], bad_letter[1])
ax2 = fig2.add_subplot(1, 1, 1)

y_des = alpha_letter
# test normal run
dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=weights_per_dimension)
y_track = []
dy_track = []
ddy_track = []

dmp.imitate_path(y_des=y_des, plot=False)
y_track, dy_track, ddy_track = dmp.rollout()

ax.plot(y_track[:, 0], y_track[:, 1],linestyle= '--', lw=1)

y_des = bad_letter
# test normal run
dmp_bad = pydmps.dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=weights_per_dimension)
dmp_bad.imitate_path(y_des=y_des, plot=False)
y_track, dy_track, ddy_track = dmp_bad.rollout()
ax.plot(y_track[:, 0], y_track[:, 1],linestyle= '--', lw=1)
ax2.plot(y_track[:, 0], y_track[:, 1],linestyle= '--', lw=1)
#ax2.plot(y_track[:, 0], y_track[:, 1],linestyle= '--', lw=1)

w = dmp.w
offset = 0
dmp2 = pydmps.dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=weights_per_dimension, w=w, goal=dmp.goal+offset, y0=dmp.y0+offset)
y_track2, dy_track2, ddy_track2 = dmp2.rollout()
ax.plot(y_track2[:, 0], y_track2[:, 1],linestyle= ':', lw=1)
ax2.plot(y_track2[:, 0], y_track2[:, 1],linestyle= ':', lw=1)
print(dmp2.y0)
print(dmp2.goal)

offset=0
dmp2 = pydmps.dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=weights_per_dimension, w=w, goal=dmp.goal+offset, y0=dmp.y0+offset)
y_track2, dy_track2, ddy_track2 = dmp2.rollout()
ax2.plot(y_track2[:, 0], y_track2[:, 1],linestyle= '--', lw=1)

w_diff = dmp_bad.w - dmp.w

plot_transition = 1
if(plot_transition):
    dmp_merge = pydmps.dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=weights_per_dimension, w=w+w_diff/4, goal=dmp.goal, y0=dmp.y0)
    y_track, dy_track, ddy_track = dmp_merge.rollout()
    ax2.plot(y_track[:, 0], y_track[:, 1],linestyle= '--', lw=1)

    dmp_merge = pydmps.dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=weights_per_dimension, w=w+w_diff/2, goal=dmp.goal, y0=dmp.y0)
    y_track, dy_track, ddy_track = dmp_merge.rollout()
    ax2.plot(y_track[:, 0], y_track[:, 1],linestyle= '--', lw=1)

    dmp_merge = pydmps.dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=weights_per_dimension, w=w+3*w_diff/4, goal=dmp.goal, y0=dmp.y0)
    y_track, dy_track, ddy_track = dmp_merge.rollout()
    ax2.plot(y_track[:, 0], y_track[:, 1],linestyle= '--', lw=1)

    dmp_merge = pydmps.dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=weights_per_dimension, w=w+w_diff, goal=dmp.goal, y0=dmp.y0)
    y_track, dy_track, ddy_track = dmp_merge.rollout()
    ax2.plot(y_track[:, 0], y_track[:, 1],linestyle= '--', lw=1)

plt.show()

print(dmp.y0)
print(dmp.goal)
print(dmp_bad.goal)
