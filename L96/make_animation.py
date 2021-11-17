import os
import numpy as np
from pyDOE import lhs
from L96 import Lorenz96
from data import generate_data

K, J = 8, 32
n_ts = 1000
# The function below creates the dataset
def create_dataset(K, J, N_ic=100, N_steps=1, dt=0.05):
    '''
    Creates either an LHS (i.e., N_steps=1) or an orbit dataset. 
    '''
    # LHS sample of 'initial conditions + parameters' (size : N_ic)
    #x_lhs_ic = lhs(6, samples=N_ic, criterion="center")*(max_bounds - min_bounds) + min_bounds
    np.random.seed(42)
    x0, y0 = np.random.random(K), np.random.random(K*J)
    # Getting data for multiple parameterizations
    L96 = Lorenz96()
    output = generate_data(L96, x0=x0, y0=y0, n_steps=N_steps, dt=dt)
    return output


#output = create_dataset(K,J,N_steps=n_ts, dt=0.01) 
#xt, yt = output['x'], output['y']

xt, yt = np.load('data/xt_truth.npz')['arr_0'], np.load('data/yt_truth.npz')['arr_0']
xt, yt = xt[:,0], yt[:,0]


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x_theta = np.linspace(0,2*np.pi,K+1)
y_theta = np.linspace(0,2*np.pi,K*J+1)-np.pi/K

new_xt = np.zeros((1000,K+1))
new_xt[:,:K] = xt
new_xt[:,-1] = xt[:,0]

new_yt = np.zeros((1000,J*K+1))
new_yt[:,:K*J] = yt
new_yt[:,-1] = yt[:,0]

fig = plt.figure()
ax = plt.subplot(111, polar=True)
ax.set_ylim(-20,20)

line = []
N = 2

for j in range(N):
    temp, = plt.plot([], [])
    line.append(temp)

line = tuple(line)

def init() :
    line[0].set_data([], [])
    line[1].set_data([], [])
    return line[0], line[1]

def update(i) :
    line[0].set_data(x_theta, new_xt[i])
    # Attention, y*5 represente dans l'animation
    line[1].set_data(y_theta, new_yt[i])
    return line[0], line[1]

anim = FuncAnimation(fig, update, init_func=init, frames=500, 
        interval=35, blit=True, repeat=False)
plt.show()
plt.close()

"""

if __name__ == '__main__':
    # Parsing arguments
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    # Experience type : 'truth', '1d' or '2d'.
    parser.add_argument('-exp', '--experience', type=str, default='2d', 
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
 
    args = parser.parse_args()

    # Truth value of Lorenz'63 parameters
    truth_sigma, truth_rho, truth_beta = 10., 28., 8/3
    
    exp         = args.experience   # 'truth, '2d' or '1d'
    extra_tag   = args.extra_tag    # specific name of the learning sample
    tag         = '-a'+str(args.learning_sample)    # kind of learning sample 
    N_ts        = args.N_ts         # MTUs (with 100 MTUs = 1 year)
    N_ic        = args.N_ic         # Number of 'initial conditions
    dt          = 0.05              # Time inicrement Delta t

    # Lorenz parameters' ranges
    min_bounds = np.array([-25., -25., 0.,  10., 26., 1.5])
    max_bounds = np.array([ 25.,  25., 50., 10., 32., 3.2])


    # Configuration of the learning sample
    # Generating 'truth' dataset (i.e., theta=(10.,28.,8/3))
    if exp == 'truth':
        n_ic = 1        # Observations are generated with 1 initial condition
        min_bounds[3:6] = np.array([truth_sigma, truth_rho, truth_beta])
        max_bounds[3:6] = np.array([truth_sigma, truth_rho, truth_beta])

        # Sampling a random initial condition
        np.random.seed(42)
        x0 = np.random.random(6)
        x0 = (max_bounds-min_bounds)*x0 + min_bounds
        min_bounds[:3] = x0[:3]
        max_bounds[:3] = x0[:3]

    # Generating learning sample with theta=(10.,28.,beta)
    if exp == '1d':
        extra_tag = extra_tag+'-1d'
        min_bounds[3:5] = np.array([truth_sigma, truth_rho])
        max_bounds[3:5] = np.array([truth_sigma, truth_rho])
    
    # Generating learning sample with theta=(10.,rho,beta)
    if exp == '2d' :
        min_bounds[3] = truth_sigma
        max_bounds[3] = truth_sigma

    # Learning sample of orbits
    if tag == '-a2' :
        N_steps = int(N_ts/dt)  # Number of timesteps in the orbit

    # LHS learning sample
    else : 
        N_steps = 1
    
    
    # Generating learning sample
    x_data, y_data = create_dataset(min_bounds, max_bounds, N_ic=N_ic, N_steps=N_steps, dt=dt)

    
    # Saving learning sample
    root = 'dataset/'
    # checking whether root directory exists
    try : os.mkdir(root)
    except : pass

    # Saving learning sample to root directory
    fname_x = root+'x_data'+tag+extra_tag+'.npz'
    fname_y = root+'y_data'+tag+extra_tag+'.npz'
    
    # 'truth' datasets are saved under another filename
    if exp == 'truth' :
        fname_x = root+'xt_truth.npz'
        fname_y = root+'yt_truth.npz'
        x_data, y_data = x_data[:,0,:3], y_data[:,0,:3]


    np.savez_compressed(fname_x, x_data)
    np.savez_compressed(fname_y, y_data)
    print(' > Dataset of size ', y_data.shape, 
            ' successfully saved to %s.'% fname)
"""
