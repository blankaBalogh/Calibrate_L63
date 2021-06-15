# 3D plot of 'truth' dataset. 

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
from pyDOE import lhs
from L63_mix import Lorenz63
from data import generate_data


fdir = '../dataset/090621/'
#xt_truth = np.load(fdir+'xt_truth.npz')['arr_0']
xt_truth = np.load(fdir+'x_data-a2.npz')['arr_0']

index = np.random.randint(100)
xt_truth = xt_truth[:,index]
print('theta : ', xt_truth[0,3:])
print('mean : ', np.mean(xt_truth, axis=0))

n_b, n_show_steps = 0, -1

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(xt_truth[n_b:n_show_steps,0], xt_truth[n_b:n_show_steps,1], 
        xt_truth[n_b:n_show_steps,2], s=1, color='darkblue')
plt.show()


