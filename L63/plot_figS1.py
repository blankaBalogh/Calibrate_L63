import numpy as np
import matplotlib.pyplot as plt

from L63_mix import Lorenz63
from data import generate_data
from pyDOE import lhs


# Initial condition
mesh_len    = 20
x, y        = np.linspace(26.,32.,mesh_len), np.linspace(1.5,3.,mesh_len)
sigmas_rhos = np.array(np.meshgrid(x, y)).T.reshape(-1,2)
Thetas_to_predict = np.copy(sigmas_rhos)

x0      = np.repeat(np.array([10.,10.,25.]), mesh_len**2).reshape(3,-1).T
ic      = np.zeros((mesh_len**2, 6))
ic[...,:3], ic[...,4:] = x0[...,:3], Thetas_to_predict
sigmas  = np.repeat(10.,mesh_len**2)
ic[...,3] = sigmas



# L63 model
print(' > L63.')
print('   Computing errors with truth function f')
n_steps_val = 20000
L63         = Lorenz63()
output      = generate_data(L63, ic, n_steps=n_steps_val, dt=0.05, compute_y=False)
xt_L63      = output['x']

# Computing std & mean values over L63 orbit.
std_L63     = np.std(xt_L63[...,:3], axis=0)
mean_L63    = np.mean(xt_L63[...,:3], axis=0)

if len(std_L63.shape) == 3 :
    std_L63  = np.mean(std_L63, axis=1)
    mean_L63 = np.mean(mean_L63, axis=1)

mean_L63    = mean_L63.reshape(mesh_len, mesh_len, 3)
std_L63     = std_L63.reshape(mesh_len, mesh_len, 3)


# Plotting layout
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10,7))

for i in range(3) :
    im=ax[0,i].pcolormesh(y,x,mean_L63[:,:,i],cmap='jet',shading='auto')
    ax[0,i].set_xlabel(r'$\beta$')
    ax[0,i].set_ylabel(r'$\rho$')
    ax[0,i].set_title(r'$\widebar{x}_{%i}$'%(i+1))
    fig.colorbar(im, ax=ax[0,i], extend='both')
    im=ax[1,i].pcolormesh(y,x,std_L63[:,:,i],cmap='jet',shading='auto')
    ax[1,i].set_xlabel(r'$\beta$')
    ax[1,i].set_ylabel(r'$\rho$')
    ax[1,i].set_title(r'$\sigma_{x_{%i}}$'%(i+1))
    fig.colorbar(im, ax=ax[1,i], extend='both')

plt.suptitle('Mean values and standard deviations : L63 model')

plt.tight_layout()
plt.show()

