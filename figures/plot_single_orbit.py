# 3d plot of any given orbit available in the dataset. 

#import matplotlib
#matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
from pyDOE import lhs
from L63_mix import Lorenz63
from data import generate_data

# Opening datasets
#fdir = 'dataset/'
#xt_pred = np.load(fdir+'x_data-a2-sigma10.0-rho28.0-beta2.2.npz')['arr_0']
#xt_pred = np.load(fdir+'xt_truth.npz')['arr_0']
#fdir = '/cnrm/amacs/USERS/baloghb/calibration_L63/v6/new_exp2/dataset/'
fdir = '../dataset/'
xt_pred = np.load(fdir+'090621/fixed_sigmas-f-obs-090621-larger.npz')['arr_0']
x0_pred = xt_pred[0]
xt_pred = np.array([[x[i*50:(i+1)*50] for i in range(100)] for x in xt_pred])
# Resulting array of shape : (20000,100,50,6).

# Computing truth orbits for comparison
print('x0 shape : ', x0_pred.shape)
n_steps = xt_pred.shape[0]

# Comparing orbits
n_orbits_comp = 10
n_b, n_show_steps = -1000, -1

#print('theta_0 : ', xt_pred[0,0,3:])

for i in [52,53,62,63] :#range(n_orbits_comp) :
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for j in range(25) :
        #ax.scatter(x0_pred[i,0], x0_pred[i,1], x0_pred[i,2], color='darkgreen', s=75, marker='+')
        ax.plot(xt_pred[n_b:n_show_steps,i,j,0], xt_pred[n_b:n_show_steps,i,j,1], 
                xt_pred[n_b:n_show_steps,i,j,2], lw=1)
    plt.show()
    #plt.savefig('figures/plots_310521/orbits_%d.png'%(i))
    plt.close()

"""
# Plotting truth observations
ind_max = 4
xt_truth = np.load('dataset/xt_truth.npz')['arr_0']
fig = plt.figure()
ax = plt.axes(projection='3d')
#ax.plot(xt_truth[:,0], xt_truth[:,1], xt_truth[:,2], lw=1)
ax.plot(xt_orb[:ind_max,4,0], xt_orb[:ind_max,4,1], xt_orb[:ind_max,4,2], lw=1)
plt.show()


plt.close()
"""


