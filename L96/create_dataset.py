import os
import numpy as np
from pyDOE import lhs
from L96 import Lorenz96
from data import generate_data

K, J = 8, 32
n_ts = 1000
# The function below creates the dataset
def create_dataset(K, J, min_bounds, max_bounds, N_ic=100, N_steps=1, dt=0.05, 
        compute_y=False, mode='standard'):
    '''
    Creates either an LHS (i.e., N_steps=1) or an orbit dataset. 
    '''
    # LHS sample of 'initial conditions + parameters' (size : N_ic)
    #x_lhs_ic = lhs(6, samples=N_ic, criterion="center")*(max_bounds - min_bounds) + min_bounds
    np.random.seed(42)
    x0 = (max_bounds-min_bounds)*np.random.random((N_ic,K+K*J+4))+min_bounds
    h,c,b,F = x0[...,-4], x0[...,-3], x0[...,-2], x0[...,-1]
    #x0,y0 = x0[...,:K],x0[...,K:K+K*J]
    
    dict_L96 = {'h':h,'c':c,'b':b,'F':F,'K':K,'J':J}
    if mode=='polynom' :
        dict_L96['mode'] = 'polynom'
        print(dict_L96)

    # Getting data for multiple parameterizations
    L96 = Lorenz96(dict_L96)
    output = generate_data(L96, x0=x0, n_steps=N_steps, dt=dt, compute_y=compute_y)
    #output['F'] = F
    #output['b'] = b
    return output


#output = create_dataset(K,J,N_steps=n_ts, dt=0.01) 
#xt, yt = output['x'], output['y']


if __name__ == '__main__':
    # Parsing arguments
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    # Experience type : 'truth', '1d' or '2d'.
    parser.add_argument('-exp', '--experience', type=str, default='1d', 
            help="exp. type : 'std', '1d', '2d'.")
    # Type of learning sample : a=1 means LHS, a=2 means 'orbit'.
    parser.add_argument('-a', '--learning_sample', type=int, default=1, 
            help='Type of learning sample.')
    # Number of initial condition (also sample size of the LHS sample).
    parser.add_argument('-n_ic', '--N_ic', type=int, default=1, 
            help='Number of initial conditions.')
    # If a=2, number of time steps in MTU (1 MTU is equivalent to 20 integrations).
    parser.add_argument('-n_ts', '--N_ts', type=int, default=100, 
            help='Number of time steps in the orbit.')
    # Extra tag that will be added to the output file name.
    parser.add_argument('-et', '--extra_tag', type=str, default='', 
            help='Extra tag to identify the obtained dataset.')
    parser.add_argument('-m', '--mode', type=str, default='lorenz',
            help='Lorenz 96 model mode (e.g., "lorenz", "polynom").')
 
    args = parser.parse_args()

    # Truth value of Lorenz'63 parameters
    h0, c0, F0, b0 = 1., 4., 20., 10.
    
    exp         = args.experience   # 'truth, '2d' or '1d'
    extra_tag   = args.extra_tag    # specific name of the learning sample
    tag         = '-a'+str(args.learning_sample)    # kind of learning sample 
    N_ts        = args.N_ts         # MTUs (with 100 MTUs = 1 year)
    N_ic        = args.N_ic         # Number of 'initial conditions
    dt          = 0.01              # Time inicrement Delta t
    mode        = args.mode         # Experience type : 'lorenz' or 'polynom'

    # Lorenz parameters' ranges
    min_bounds = np.array([-1]*K+[-1]*(K*J)+[8.,0.5,8.,8.])
    max_bounds = np.array([1]*K+[1]*(K*J)+[12.,1.5,12.,12.])

    # Configuration of the learning sample
    # Generating 'truth' dataset (i.e., theta=(10.,28.,8/3))
    if exp == 'truth':
        n_ic = 1        # Observations are generated with 1 initial condition
        min_bounds[-4:] = np.array([h0,c0,F0,b0])
        max_bounds[-4:] = np.array([h0,c0,F0,b0])

        # Sampling random initial condition
        np.random.seed(42)
        x0 = np.random.random(K+K*J+4)
        x0 = (max_bounds-min_bounds)*x0 + min_bounds

    # Generating learning sample with theta=(10.,28.,beta)
    if exp == '1d':
        extra_tag = extra_tag+'-1d'
        min_bounds[-4:-1] = np.array([h0,c0,F0])
        max_bounds[-4:-1] = np.array([h0,c0,F0])
    
    # Generating learning sample with theta=(10.,rho,beta)
    if exp == '2d' :
        min_bounds[-4:-2] = np.array([h0,c0])
        max_bounds[-4:-2] = np.array([h0,c0])

    # Learning sample of orbits 
    if tag == '-a2' :
        N_steps = N_ts  # Number of timesteps in the orbit

    # LHS learning sample
    else : 
        N_steps = 1
     
    # Generating learning sample
    compute_y = True
    output = create_dataset(K, J, min_bounds, max_bounds, N_ic=N_ic, N_steps=N_steps, dt=dt,
            compute_y=compute_y, mode=mode)
    x_data, y_data = output['x'], output['y']
    F, b = output['F'], output['b']
    dx, dy = output['dx'], output['dy']

    # Saving learning sample
    root = 'data/'
    # checking whether root directory exists
    try : os.mkdir(root)
    except : pass

    # Saving learning sample to root directory
    fname_x = root+'x_data'+tag+extra_tag+'.npz'
    fname_y = root+'y_data'+tag+extra_tag+'.npz'
    if exp=='1d' :
        fname_b = root+'b_data'+tag+extra_tag+'.npz'
    elif exp=='2d' :
        fname_F = root+'F_data'+tag+extra_tag+'.npz'
        fname_b = root+'b_data'+tag+extra_tag+'.npz'

    if compute_y :
        fname_dx= root+'dx_data'+tag+extra_tag+'.npz'
        fname_dy= root+'dy_data'+tag+extra_tag+'.npz'


    # 'truth' datasets are saved under another filename
    if exp == 'truth' :
        fname_x = root+'xt_truth.npz'
        fname_y = root+'yt_truth.npz'
        if mode == 'polynom' :
            fname_x, fname_y = fname_x+'-polynom', fname_y+'-polynom'
        #x_data, y_data = x_data[:,0,:3], y_data[:,0,:3]
    
    print('x_data shape : ', x_data.shape)
    np.savez_compressed(fname_x, x_data)
    np.savez_compressed(fname_y, y_data)

    if exp=='1d' :
        np.savez_compressed(fname_b, b)

        if compute_y :
            np.savez_compressed(fname_dx, dx)
            np.savez_compressed(fname_dy, dy)

    elif exp=='2d' :
        np.savez_compressed(fname_F, F)
        np.savez_compressed(fname_b, b)
        
        if compute_y :
            np.savez_compressed(fname_dx, dx)
            np.savez_compressed(fname_dy, dy)

    print(' > Dataset of size ', y_data.shape, 
            ' successfully saved to %s.'% fname_y)
