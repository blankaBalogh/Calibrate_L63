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

    nx, nth = 500, 100
    
    # Generating x variables
    x0_min, x0_max = np.array([-25., -25., 0.]), np.array([25., 25., 50.])
    delta_x0 = x0_max - x0_min
    x0_var = lhs(3, samples=nx)*delta_x0 + x0_min

    # Generating theta variables
    th_min, th_max = np.array([9.,26.5,1.5]), np.array([11.,29.5,3.])
    delta_th = th_max - th_min
    th_var = lhs(3, samples=nth)*delta_th + th_min

    # Combination of x & theta variables
    x_data = np.zeros((nx*nth,6))*np.nan
    x_data[:,:3] = np.repeat(x0_var,nth).reshape(3,nx*nth).T
    x_data[:,3:] = np.tile(th_var,(nx,1))

    #x_data = np.stack(np.meshgrid(x0_var, th_var)).T.reshape(-1,6)

    y_data = np.zeros((nx*nth,6))
    y_data[:,0] = x_data[:,3]*(x_data[:,1] - x_data[:,0])
    y_data[:,1] = x_data[:,0]*(x_data[:,4]-x_data[:,2]) - x_data[:,1]
    y_data[:,2] = x_data[:,1]*x_data[:,0] - x_data[:,5]*x_data[:,2]

    x_data, y_data = x_data.reshape(1,nx*nth,6), y_data.reshape(1,nx*nth,6)

    root = 'dataset/'
    tag, extra_tag = '-a1', '-5e4'
    fname = root+'x_data'+tag+extra_tag+'.npz'
    np.savez_compressed(fname, x_data)
    np.savez_compressed(root+'y_data'+tag+extra_tag+'.npz', y_data)
    print(' > Dataset of size ', y_data.shape, 
            ' successfully saved to %s.'% fname)
