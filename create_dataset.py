import numpy as np
import pandas as pd
from pyDOE import lhs
from L63_mix import Lorenz63
from data import generate_data


def create_dataset(min_bounds, max_bounds, N_ic=100, N_steps=1, dt=0.05):
    '''
    '''
    # LHS sample of 'initial conditions + parameters' (size : N_ic)
    lhs_ic = lhs(6, samples=N_ic, criterion="center")*(max_bounds - min_bounds) + min_bounds

    # Getting data for multiple parameterizations
    L63 = Lorenz63()
    output = generate_data(L63, x0=lhs_ic, n_steps=N_steps, dt=dt, compute_y=True)

    return output['x'], output['y']


if __name__ == '__main__':
    # Parsing arguments
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-exp', '--experience', type=str, default='3d', 
            help="exp. type : 'std', '1d' or '3d'.")
    parser.add_argument('-1ts', '--ls_1ts', default=False, action='store_true',
            help='Enables the creation of an 1ts learning sample.')
    
    args = parser.parse_args()
    
    truth_sigma, truth_rho, truth_beta = 10., 28., 8/3
    
    exp = args.experience # 'std' or '1d'
    l1ts = args.ls_1ts
    
    N_ic = 100  # Number of 'initial conditions + parameters' to sample
    min_bounds = np.array([-10, -10, 10,  9, 26.5, 1.5])
    max_bounds = np.array([ 10,  10, 30, 10, 29.5, 3.0])
    
    if exp == 'std':
        min_bounds[3:6] = np.array([truth_sigma, truth_rho, truth_beta])
        max_bounds[3:6] = np.array([truth_sigma, truth_rho, truth_beta])

    if exp == '1d':
        min_bounds[3:5] = np.array([truth_sigma, truth_rho])
        max_bounds[3:5] = np.array([truth_sigma, truth_rho])
        
    dt = 0.05
    N_ts = 100 # MTUs (with 100 MTUs = 1 year)
    if not l1ts:
        N_steps = int(N_ts/dt)  # Number of timesteps in the orbit
        tag = '-a2'
    else :
        N_ic = 25000 
        N_steps = 1
        tag = '-a1'
        
    x_data, y_data = create_dataset(min_bounds, max_bounds, N_ic=N_ic, N_steps=N_steps, dt=dt)

    root = 'dataset/'
    fname = root+'x_data'+tag+'.npz'
    np.savez_compressed(fname, x_data)
    np.savez_compressed(root+'y_data'+tag+'.npz', y_data)
    print(' > Dataset of size ', y_data.shape, 
            ' successfully saved to %s.'% fname)
