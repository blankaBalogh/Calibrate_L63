# Calibration tool for L63 model.

# -------------------------------------------------------------------------------- #
# -------------------------------    Importations   ------------------------------ #
# -------------------------------------------------------------------------------- #
import matplotlib
matplotlib.use('Agg')

import os
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
import GPy

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus :
    tf.config.experimental.set_memory_growth(gpu, True)

import matplotlib.pyplot as plt


# -- Parsing aruments
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-a', '--learning_sample', default=2,
        help='Learning sample selection : orbit (=2) or lhs (=1).')
parser.add_argument('-et', '--extra_tag', type=str, default='', 
        help='Adds an extra tag. Useful to save new datasets.')
parser.add_argument('-gp', '--new_gp', default=False, action='store_true')

args    = parser.parse_args()

tag = '-a'+str(args.learning_sample)
#tag_m = '-m'+str(args.metric)
#tag_o = '-o'+str(args.optimisation)
extra_tag = args.extra_tag

#new_gp_ls = args.new_gp_ls

if tag=='-a1' : learning_sample = 'LHS'
else : learning_sample = 'orbits'

new_gp = args.new_gp

"""
if tag_m=='-m1' : metric = 'mean+std'
else : metric = 'std'


if tag_o=='-o1' : optim = 'kriging'
else : optim = 'raw'
"""

print()
print('> Learning sample : %s.'%learning_sample)
#print('> Metric : %s.'%metric)
#print('> Optimizer : %s.'%optim)
#if new_gp_ls :
#    print('> New GP learning sample.')


# -------------------------------------------------------------------------------- #
# ---------------------    Loading available 'observations'   -------------------- #
# -------------------------------------------------------------------------------- #

# 'Observations' can be loaded from the 'datatset' directory.  
# They are then used to calculate longterm metric (e.g., mean, std, covaX0 = np.array([x0 for i in range(mesh_len**2)]).reshape(-1,3)riance)
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

if tag=='-amix' :
    print(' > Loading mixed learning sample.')
    x_data = np.load('dataset/x_data-amix'+extra_tag+'.npz')['arr_0']
    y_data = np.load('dataset/y_data-amix'+extra_tag+'.npz')['arr_0'][...,:3]


# --- Learning fhat_betas
print('\n ------ Learning fhat_thetas ------- ')
# Normalization of x & y data
mean_x, std_x = np.mean(x_data, axis=0), np.std(x_data, axis=0)
mean_y, std_y = np.mean(y_data, axis=0), np.std(y_data, axis=0)
x_data = (x_data-mean_x)/std_x
y_data = (y_data-mean_y)/std_y

# Setting up NN model
#layers = [1024, 512, 256, 128, 64, 32, 16]
#layers = [64, 32, 16]
layers = [256, 128, 64, 32, 16]
#layers = [256, 256, 256, 256, 128, 64, 32, 16]

print('y data shape : ', y_data.shape)

dic_NN = {'name':'f_orb', 'in_dim':x_data.shape[1], 'out_dim':y_data.shape[1], 
        'nlays':layers}
nn_L63 = ML_model(dic_NN)
nn_L63.norms = [mean_x, mean_y, std_x, std_y]
#extra_tag = extra_tag+'-largerNN'
extra_tag = extra_tag
nn_L63.suffix = tag+extra_tag
nn_L63.name = 'model_'+tag+extra_tag
print(nn_L63.model.summary())

print(' > Loading model weights.')
nn_L63.model.load_weights('weights/weights'+nn_L63.suffix+'.h5')

'''
if tag_m == '-m2' :
    alpha=1.0
else :
    alpha=0.5
'''

n_steps_loss, n_snapshots = 200, 50

x0 = np.zeros((n_snapshots, 6))
index_valid = np.random.randint(0, xt_truth.shape[0]-1, n_snapshots)
x0[...,:3] = xt_truth[index_valid]
#x0[...,3:] = np.repeat([10.,28.,8/3], 50).reshape(3, n_snapshots).T

dt = 0.05

"""
comp_loss_data = compute_loss_data(nn_L63, xt_truth, x0=x0, n_steps=n_steps_loss, 
        dt=0.05, alpha=alpha, tag=tag, extra_tag=extra_tag)
"""
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

n_snapshots, n_thetas = 50, 150
n_steps = 20000
sdir = 'dataset/070621/'
try : os.mkdir(sdir)
except : pass

if new_gp :
    print(' > Computing new learning sample for kriging.')
    index_valid = np.random.randint(0, xt_truth.shape[0]-1, n_snapshots)
    x0 = xt_truth[index_valid]

    min_bounds_Th, max_bounds_Th = np.array([10.,26.5,1.5]), np.array([10.,32.,3.2])
    thetas_list = lhs(3, samples=n_thetas)*(max_bounds_Th-min_bounds_Th)+min_bounds_Th
    Thetas = np.repeat(thetas_list, n_snapshots).reshape(3,-1).T

    X0 = np.array([x0 for i in range(n_thetas)]).reshape(-1,3)

    ic = np.zeros((n_thetas*n_snapshots,6))
    ic[:,:3], ic[:,3:] = X0, Thetas

    print(' > Computing output')
    
    L63 = Lorenz63()
    output = generate_data(L63, ic, n_steps=n_steps, dt=0.05, compute_y=False)
    xt_pred = output['x']
    np.savez_compressed(sdir+'raw_xt-obs.npz', xt_pred)

    xt_pred = np.array([[x[i*n_snapshots:(i+1)*n_snapshots] for i in range(n_thetas)] \
        for x in xt_pred])


    # Computing errors
    # Computing predicted orbits standard deviation
    mean_pred = np.mean(xt_pred, axis=(0))[:,:,:3]
    mean_pred = np.mean(mean_pred, axis=1)

    std_pred = np.std(xt_pred, axis=0)[:,:,:3]
    std_pred = np.mean(std_pred, axis=1)

    # Computing truth standard deviation
    mean_truth = np.mean(xt_truth, axis=0)
    std_truth = np.std(xt_truth, axis=0)

    # Computing error (MSE) on STDs
    err_mean = np.mean((mean_pred-mean_truth)**2, axis=1)
    err_std = np.mean((std_pred-std_truth)**2, axis=1)

    saving_gp = True  
    extra_tag = extra_tag
    if saving_gp :
        print(' > Saving GP learning sample (STD).')
        np.savez_compressed(sdir+'train_orbits_gp'+tag+extra_tag+'.npz',
                xt_pred)
        np.savez_compressed(sdir+'train_thetas_gp'+tag+extra_tag+'.npz', 
                thetas_list)
        np.savez_compressed(sdir+'train_errors_std_gp'+tag+extra_tag+'.npz',
                err_std)
        np.savez_compressed(sdir+'train_errors_mean_gp'+tag+extra_tag+'.npz', 
                err_mean)

else :
    print(' > Loading learning sample for kriging.')
    err_std = np.load(sdir+'train_errors_std_gp'+tag+extra_tag+'.npz')['arr_0']
    err_mean = np.load(sdir+'train_errors_mean_gp'+tag+extra_tag+'.npz')['arr_0']
    thetas_list = np.load(sdir+'train_thetas_gp'+tag+extra_tag+'.npz')['arr_0']


## Calibration
err_type='mean+std'
if err_type=='std' :
    y = np.copy(err_std)
else :
    y = err_std+err_mean
thetas_list = thetas_list
print(thetas_list.shape)
mean_thetas, std_thetas = np.mean(thetas_list, axis=0), np.std(thetas_list, axis=0)
mean_y, std_y = np.mean(err_std, axis=0), np.std(err_std, axis=0)

norm_gp = True
if norm_gp :
    thetas_list = (thetas_list-mean_thetas)/std_thetas
    y = (y-mean_y)/std_y

y = err_std.reshape(-1,1)

thetas_list = thetas_list.reshape(-1,3)
kernel = GPy.kern.Matern52(input_dim=3, ARD=True)
gp = GPy.models.GPRegression(thetas_list, y, kernel)
gp.optimize(messages=True)
print(gp)
print(gp.kern.lengthscale)

print('\n -------  Optimization  -------')

norms_gp = np.array([mean_thetas, mean_y, std_thetas, std_y])

loss_kriging = compute_loss_kriging(gp, norm_gp=norm_gp, norms=norms_gp)
    
theta_to_update = [9.5, 28.5, 2.5]
print(' > initial theta value : ', theta_to_update)

eps, tol = 1e-1, 1e-2

res = minimize(loss_kriging, theta_to_update, method='BFGS', tol=tol, 
        callback=callbackF, options={'eps':eps})
    
theta_star = res.x

print(' > optimal theta value : ', theta_star)


### END OF SCRIPT ###
print(' > Done.')
exit()
