import numpy as np
import pandas as pd
import os
from pyDOE import lhs
from L63_mix import Lorenz63
from data import generate_data

beta0 = 8/3
xt_truth = np.load('dataset/xt_truth.npz')['arr_0']
#xt_truth = np.load('dataset/xt_truth-sigma10.0-rho28.0-beta'+str(beta0)+'.npz')['arr_0']
if len(xt_truth.shape)==3 :
    xt_truth = xt_truth[:,0,:3]

x0 = np.zeros(6)
x0[:3] = xt_truth[0]

def create_dataset(x0, N_ic=100, N_steps=1, dt=0.05):
    '''
    '''
    # Getting data for multiple parameterizations
    L63 = Lorenz63()
    output = generate_data(L63, x0=x0, n_steps=N_steps, dt=dt, compute_y=True)
    return output['x']


dt, N_ts = 0.05, 3000
N_steps = int(N_ts/dt)  # Number of timesteps in the orbit
    
N_ic = 1  # Number of 'initial conditions + parameters' to sample
xt_to_save = np.zeros((N_steps, 100, 6))
    
count = 0

sigma0, rho0 = 10., 28.

for sigma in [sigma0] : #np.linspace(7.,13.,10) :
    for rho in np.linspace(26.5,32.,10) :
        for beta in np.linspace(1.5,3.2,10) :
            x0[3:] = [sigma, rho, beta]   
            x_data = create_dataset(x0, N_ic=N_ic, N_steps=N_steps, dt=dt)
            xt_to_save[:,count] = x_data[:,0]

            count += 1

root = 'dataset/'
try : os.mkdir(root)
except : pass

fname = root+'x_data-f-rho-beta-sigma10-larger.npz'
np.savez_compressed(fname, xt_to_save)
print(' > Dataset successfully saved to %s.'% fname)
