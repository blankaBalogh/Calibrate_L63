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
fdir = '/cnrm/amacs/USERS/baloghb/calibration_L63/v6/new_exp2/dataset/'
#fdir = 'dataset/'
xt_pred = np.load(fdir+'090621/fixed_betas-f-pred-090621.npz')['arr_0']
x0_pred = xt_pred[0]
xt_pred = np.array([[x[i*50:(i+1)*50] for i in range(100)] for x in xt_pred])
# Resulting array of shape : (20000,100,50,6).

# Computing truth orbits for comparison
print('x0 shape : ', x0_pred.shape)
n_steps = xt_pred.shape[0]


L63 = Lorenz63()
output = generate_data(L63, x0=x0_pred, n_steps=n_steps, dt=0.01, compute_y=False)
xt_obs = output['x']
xt_obs = np.array([[x[i*50:(i+1)*50] for i in range(100)] for x in xt_obs])
#xt_obs = np.load('dataset/x_data-a2.npz')['arr_0']

# Comparing orbits
n_orbits_comp = 10
n_b, n_show_steps = -1000, -1

#print('theta_0 : ', xt_pred[0,0,3:])

for i in np.arange(0,10)*10 :#range(n_orbits_comp) :
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #ax.scatter(x0_pred[i,0], x0_pred[i,1], x0_pred[i,2], color='darkgreen', s=75, marker='+')
    ax.scatter(xt_pred[n_b:n_show_steps,i,0,0], xt_pred[n_b:n_show_steps,i,0,1], 
            xt_pred[n_b:n_show_steps,i,0,2], s=1, color='darkblue')
    ax.plot(xt_obs[n_b:n_show_steps,i,0,0], xt_obs[n_b:n_show_steps,i,0,1], 
            xt_obs[n_b:n_show_steps,i,0,2], lw=1, color='red')
    plt.show()
    #plt.savefig('figures/plots_310521/orbits_%d.png'%(i))
    plt.close()


n_b, n_show_steps = -5000, -1
i = 12
fig = plt.figure()
ax = plt.axes(projection='3d')
#ax.scatter(x0_pred[i,0], x0_pred[i,1], x0_pred[i,2], color='darkgreen', s=75, marker='+')
ax.scatter(xt_pred[n_b:n_show_steps,i,0,0], xfixed_sigmas-f-pred.npzt_pred[n_b:n_show_steps,i,0,1], 
        xt_pred[n_b:n_show_steps,i,0,2], s=1, color='darkblue')
ax.plot(xt_obs[n_b:n_show_steps,i,0,0], xt_obs[n_b:n_show_steps,i,0,1], 
        xt_obs[n_b:n_show_steps,i,0,2], lw=1, color='red')
plt.show()
#plt.savefig('figures/plots_310521/orbits_%d.png'%(i))
plt.close()

j = 19
n_b, n_show_steps = 0, 5000

for i in range(10) :
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #ax.scatter(x0_pred[i,0], x0_pred[i,1], x0_pred[i,2], color='darkgreen', s=75, marker='+')
    ax.scatter(xt_L63[n_b:n_show_steps,i,0,0], xt_L63[n_b:n_show_steps,i,0,1], 
            xt_L63[n_b:n_show_steps,i,0,2], s=1, color='darkblue')
    ax.plot(xt_nn[n_b:n_show_steps,j,i,0], xt_nn[n_b:n_show_steps,j,i,1], 
            xt_nn[n_b:n_show_steps,j,i,2], lw=1, color='red')
    plt.show()
    plt.close()

n_b, n_show_steps = -5000, -1

for i in range(10) :
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #ax.scatter(x0_pred[i,0], x0_pred[i,1], x0_pred[i,2], color='darkgreen', s=75, marker='+')
    #ax.scatter(x_5e4[n_b:n_show_steps,i,0,0], x_5e4[n_b:n_show_steps,i,0,1], 
    #        x_5e4[n_b:n_show_steps,i,0,2], s=1, color='darkblue')
    ax.plot(xt[n_b:n_show_steps,i,0], xt[n_b:n_show_steps,i,1], 
            xt[n_b:n_show_steps,i,2], lw=1, color='red')
    plt.show()
    plt.close()




n_b, n_show_steps = -5000, -1

fig = plt.figure()
ax = plt.axes(projection='3d')
#ax.scatter(x0_pred[i,0], x0_pred[i,1], x0_pred[i,2], color='darkgreen', s=75, marker='+')
#ax.scatter(x_5e4[n_b:n_show_steps,i,0,0], x_5e4[n_b:n_show_steps,i,0,1], 
#        x_5e4[n_b:n_show_steps,i,0,2], s=1, color='darkblue')
ax.plot(xt[n_b:n_show_steps,0], xt[n_b:n_show_steps,1], xt[n_b:n_show_steps,2], 
        lw=1, color='red')
plt.show()
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


