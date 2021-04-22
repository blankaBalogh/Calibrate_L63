# Calibration tool for L63 model.

# -------------------------------------------------------------------------------- #
# -------------------------------    Importations   ------------------------------ #
# -------------------------------------------------------------------------------- #
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import time
import seaborn as sns
sns.set_style('white')
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

import matplotlib.pyplot as plt


# -- Parsing aruments
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
layers = [64, 32, 16]

print('y data shape : ', y_data.shape)

dic_NN = {'name':'f_orb', 'in_dim':x_data.shape[1], 'out_dim':y_data.shape[1], 
        'nlays':layers}
nn_L63 = ML_model(dic_NN)
nn_L63.norms = [mean_x, mean_y, std_x, std_y]
nn_L63.suffix = tag+extra_tag
print(nn_L63.model.summary())

print(' > Loading model weights.')
nn_L63.model.load_weights('weights/weights'+nn_L63.suffix+'.h5')


if tag_m == '-m2' :
    alpha=1.0
else :
    alpha=0.5

n_steps_loss, n_snapshots = 200, 50

x0 = np.zeros((n_snapshots, 6))
index_valid = np.random.randint(0, xt_truth.shape[0]-1, n_snapshots)
x0[...,:3] = xt_truth[index_valid]
#x0[...,3:] = np.repeat([10.,28.,8/3], 50).reshape(3, n_snapshots).T

dt = 0.05

comp_loss_data = compute_loss_data(nn_L63, xt_truth, x0=x0, n_steps=n_steps_loss, 
        dt=0.05, alpha=alpha, tag=tag, extra_tag=extra_tag)
n_iter = 0

def callbackF(x_) :
    '''
    Function to print all intermediate values
    '''
    global n_iter
    if n_iter%10 == 0 :
        print("Iteration no.", n_iter)
        print("theta value : ", x_)
    n_iter += 1


if tag_o=='-o1' :
    print('\n ------- Defining and fitting GP regressor ------- ')
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern

    n_samples = 150
    min_bounds, max_bounds = np.array([9.,26.5,2.]), np.array([11., 29., 3.])
    thetas_list = lhs(3, samples=n_samples)*(max_bounds-min_bounds)+min_bounds
    saving_gp = True

    if new_gp_ls :
        errors = np.zeros((n_samples, 1)) * np.nan

        for (i,theta) in enumerate(thetas_list) :
            print(' > err. no.%d/%d.' % ((i+1), n_samples))
            errors[i,0] = comp_loss_data(theta, i=i)

        if saving_gp :
            print(' > Saving GP learning sample.')
            np.savez_compressed('dataset/thetas_errors/train_thetas_gp'+tag+tag_m+ \
                    extra_tag+'.npz', thetas_list)
            np.savez_compressed('dataset/thetas_errors/train_errors_gp'+tag+tag_m+ \
                    extra_tag+'.npz', errors)

    else :
        thetas_list = np.load('dataset/thetas_errors/train_thetas_gp'+tag+tag_m+\
                extra_tag+'.npz')['arr_0']
        errors = np.load('dataset/thetas_errors/train_errors_gp'+tag+tag_m+ \
                extra_tag+'.npz')['arr_0']

    # fitting GP 
    kernel = Matern(length_scale=1.0, nu=1.5)       # Definition of Matern kernel
    gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
    thetas_list = thetas_list.reshape(-1,3)
    gp.fit(thetas_list, errors)

    print('\n -------  Optimization  -------')

    loss_kriging = compute_loss_kriging(gp)

    theta_to_update = [9.5, 28.5, 2.1]
    print(' > initial theta value : ', theta_to_update)
    res = minimize(loss_kriging, theta_to_update, method='BFGS', tol=1e-1, 
            callback=callbackF, options={'eps':1e-2})

    print(' > optimal theta value : ', res.x)

else :
    print('\n ------- Optimization on raw data ------- ')
    loss_fun = compute_loss_data(nn_L63, xt_truth, x0=x0, n_steps=n_steps_loss, 
            dt=0.05, alpha=alpha)
    theta_to_update = [9.5, 28.5, 2.1]
    print(' > initial theta value : ', theta_to_update)
    res = minimize(loss_fun, theta_to_update, method='BFGS', tol=1e-1, 
            callback=callbackF, options={'eps':1e-2})
    print(' > optimal theta value : ', res.x)





# Generation of a validation orbit
gen_valid_orbit = False

if gen_valid_orbit :
    x0 = xt_truth[0]
    # Generation of a (long) validation orbit 
    print(' > Generating a validation orbit...')
    val_orb = generate_data(nn_L63, x0, thetas=res.x, n_steps=60000, dt=dt,
            compute_y=False, sigmai=10., rhoi=28.)
    xt_valid = val_orb['x']
    np.savez_compressed('dataset/validation_orbit_kriging-'+learning_sample+'.npz', 
            xt_valid)

print(' > Done.')
exit()
