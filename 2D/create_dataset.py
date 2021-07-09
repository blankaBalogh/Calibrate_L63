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
            help="exp. type : 'std', '1d', '3d'.")
    parser.add_argument('-a', '--learning_sample', type=int, default=1, 
            help='Type of learning sample.')
    parser.add_argument('-n_ic', '--N_ic', type=int, default=1, 
            help='Number of initial conditions.')
    parser.add_argument('-n_ts', '--N_ts', type=int, default=100, 
            help='Number of time steps in the orbit.')
    parser.add_argument('-et', '--extra_tag', type=str, default='', 
            help='Extra tag to identify the obtained dataset.')

    
    args = parser.parse_args()
    
    truth_sigma, truth_rho, truth_beta = 10., 28., 8/3
    
    exp = args.experience #'std' or '1d'
    extra_tag = args.extra_tag
    tag = '-a'+str(args.learning_sample)
    N_ts = args.N_ts  # MTUs (with 100 MTUs = 1 year)
    N_ic = args.N_ic  # Number of 'initial conditions + parameters' to sample
   
    min_bounds = np.array([-35., -35., 0.,  7., 26.5, 1.5])
    max_bounds = np.array([ 35.,  35., 60., 13., 32., 3.2])

    
    if exp == 'std':
        n_ic = 1
        min_bounds[3:6] = np.array([truth_sigma, truth_rho, truth_beta])
        max_bounds[3:6] = np.array([truth_sigma, truth_rho, truth_beta])
        np.random.seed(42)
        x0 = np.random.random(6)
        x0 = (max_bounds-min_bounds)*x0 + min_bounds
        min_bounds[:3] = x0[:3]
        max_bounds[:3] = x0[:3]

    if exp == '1d':
        extra_tag = extra_tag+'-1d'
        min_bounds[3:5] = np.array([truth_sigma, truth_rho])
        max_bounds[3:5] = np.array([truth_sigma, truth_rho])

    if exp == '2d' :
        extra_tag = extra_tag+'-2d'
        min_bounds[3] = truth_sigma
        max_bounds[3] = truth_sigma

    if exp=='mix' : 
        tag = '-a2'
        
    dt = 0.05
    if tag=='-a2' :
        N_steps = int(N_ts/dt)  # Number of timesteps in the orbit

    else : 
        N_steps = 1
        
    x_data, y_data = create_dataset(min_bounds, max_bounds, N_ic=N_ic, N_steps=N_steps, dt=dt)

    if exp=='mix' :
        x_data_lhs, y_data_lhs = create_dataset(min_bounds, max_bounds, N_ic=N_ic*N_steps, 
                N_steps=1, dt=dt)

        x_data = np.concatenate((x_data, x_data_lhs), axis=0)
        y_data = np.concatenate((y_data, y_data_lhs), axis=0)
        tag='-mix'

    root = 'dataset/'
    fname = root+'x_data'+tag+extra_tag+'.npz'
    np.savez_compressed(fname, x_data)
    np.savez_compressed(root+'y_data'+tag+extra_tag+'.npz', y_data)
    print(' > Dataset of size ', y_data.shape, 
            ' successfully saved to %s.'% fname)
