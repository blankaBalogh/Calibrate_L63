import numpy as np
from scipy.optimize import minimize
from pyDOE import lhs

#from eL63 import embeddedLorenz63
from L63_mix import Lorenz63
from ML_model_param import ML_model, train_ML_model
from data import generate_data, generate_LHS, generate_data_solvers, generate_x0 
from metrics import *

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus :
    tf.config.experimental.set_memory_growth(gpu, True)


# Parsing arguments
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-a', '--learning_sample', type=int, default=2,
        help='Learning sample selection : orbit (=2) or lhs (=1).')
parser.add_argument('-m', '--metric', type=int, default=2,
        help='Metric selection : L2 over mean+std (=1) or std only (=2).')
parser.add_argument('-o', '--optimisation', type=int, default=1, 
        help='Optimization selection : kriging (=1) or raw (=2).')
parser.add_argument('-gp', '--new_gp_ls', default=False, 
        action='store_true', help='New GP LS or not ?')
parser.add_argument('-et', '--extra_tag', type=str, default='', 
        help='Adds an extra tag. Useful to save new datasets.')

args    = parser.parse_args()

tag = '-a'+str(args.learning_sample)
tag_m = '-m'+str(args.metric)
tag_o = '-o'+str(args.optimisation)
extra_tag = args.extra_tag

new_gp_ls = args.new_gp_ls

if tag=='-a1' : learning_sample = 'LHS'
else : learning_sample = 'orbits'

if tag_m=='-m1' : metric = 'mean+std'
else : metric = 'std'

if tag_o=='-o1' : optim = 'kriging'
else : optim = 'raw'

print()
print('> Learning sample : %s.'%learning_sample)
print('> Metric : %s.'%metric)
print('> Optimizer : %s.'%optim)
if new_gp_ls :
    print('> New GP learning sample.')



# -------------------------------------------------------------------------------- #
# ---------------------    Loading available 'observations'   -------------------- #
# -------------------------------------------------------------------------------- #

# 'Observations' can be loaded from the 'datatset' directory.  
# They are then used to calculate longterm metric (e.g., mean, std, covariance)
xt_truth = np.load('dataset/xt_truth.npz')['arr_0']
yt_truth = np.load('dataset/yt_truth.npz')['arr_0']



# -------------------------------------------------------------------------------- #
# ----------------    2nd EXPERIMENT : LEARNING dx = f(x,theta)   ---------------- #
# -------------------------------------------------------------------------------- #
print('***** 2nd experiment : learning to predict full y_data. ***** ')
print(' ------- Loading learning samples ------- ')

"""
dt = 0.05
N_ic =  100             # Number of initial conditions to sample
N_ts = 100              # 200 ts with dt=0.05 is the equivalent of 1 year of data 
N_steps = int(N_ts/dt)  # Number of timesteps in the orbit
truth_sigma, truth_rho = 10., 28.
"""

# Loading 'orbits' learning sample
if tag=='-a2' :
    print(' > Loading learning sample of orbits.')
    x_data = np.load('dataset/x_data-a2'+extra_tag+'.npz')['arr_0']
    y_data = np.load('dataset/y_data-a2'+extra_tag+'.npz')['arr_0'][...,:3]
    x_data, y_data = np.swapaxes(x_data,0,1), np.swapaxes(y_data,0,1) 
    x_data = x_data.reshape(-1, x_data.shape[-1])
    y_data = y_data.reshape(-1, y_data.shape[-1])

# Loading lhs learning sample
if tag=='-a1' :
    print(' > Loading learning sample of LHS sample.')
    x_data = np.load('dataset/x_data-a1'+extra_tag+'.npz')['arr_0'][0]
    y_data = np.load('dataset/y_data-a1'+extra_tag+'.npz')['arr_0'][0][...,:3]


# --- Learning fhat_betas
print('\n ------ Learning fhat_thetas ------- ')
# Normalization of x & y data
mean_x, std_x = np.mean(x_data, axis=0), np.std(x_data, axis=0)
mean_y, std_y = np.mean(y_data, axis=0), np.std(y_data, axis=0)
x_data = (x_data-mean_x)/std_x
y_data = (y_data-mean_y)/std_y

# Setting up NN model
layers = [1024, 512, 256, 128, 64, 32, 16]

print('y data shape : ', y_data.shape)

dic_NN = {'name':'f_orb', 'in_dim':x_data.shape[1], 'out_dim':y_data.shape[1], 
        'nlays':layers}
nn_L63 = ML_model(dic_NN)
nn_L63.norms = [mean_x, mean_y, std_x, std_y]
extra_tag = extra_tag+'-largerNN'
nn_L63.suffix = tag+extra_tag
nn_L63.name = 'model_'+tag+extra_tag
print(nn_L63.model.summary())

print(' > Loading model weights.')
nn_L63.model.load_weights('weights/weights'+nn_L63.suffix+'.h5')



######
# Generating an orbit from a random initial condition

n_snapshots=50
dt = 0.05
x0 = np.zeros((n_snapshots, 6))*np.nan
index_valid = np.random.randint(0, xt_truth.shape[0]-1, n_snapshots)
x0[...,:3] = xt_truth[index_valid]
x0[...,3:] = np.repeat([10.,28.,8/3], n_snapshots).reshape(3,-1).T

val_orb = generate_data(nn_L63, x0, n_steps=20000, dt=dt, compute_y=False)

np.savez_compressed('dataset/valid_orbits/valOrb'+tag+extra_tag+'-truthTheta.npz', 
        val_orb['x'])
np.savez_compressed('dataset/valid_orbits/x0'+tag+extra_tag+'-truthTheta.npz',
        x0)
print(' > Done.')
exit()

