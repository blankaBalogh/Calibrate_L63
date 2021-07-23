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
from datetime import datetime

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
parser.add_argument('-a', '--learning_sample', default=2,
        help='Learning sample selection : orbit (=2) or lhs (=1).')
parser.add_argument('-et', '--extra_tag', type=str, default='', 
        help='Adds an extra tag. Useful to save new datasets.')

args    = parser.parse_args()

tag = '-a'+str(args.learning_sample)
#tag_m = '-m'+str(args.metric)
#tag_o = '-o'+str(args.optimisation)
extra_tag = args.extra_tag

#new_gp_ls = args.new_gp_ls

if tag=='-a1' : learning_sample = 'LHS'
else : learning_sample = 'orbits'

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
    x_data = np.delete(x_data, 3, axis=-1)

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
#layers = [256, 128, 64, 32, 16]
#layers = [256, 256, 256, 128, 64, 32, 16]
layers = [1024, 512, 256, 128, 64, 32, 16]


print('y data shape : ', y_data.shape)

dic_NN = {'name':'f_orb', 'in_dim':x_data.shape[1], 'out_dim':y_data.shape[1], 
        'nlays':layers}
nn_L63 = ML_model(dic_NN)
nn_L63.norms = [mean_x, mean_y, std_x, std_y]
#extra_tag = extra_tag+'-largerNN'
extra_tag = extra_tag+'-7dl'
nn_L63.suffix = tag+extra_tag
nn_L63.name = 'model_'+tag+extra_tag
print('> Model to load : %s.'%nn_L63.name)
print(nn_L63.model.summary())

print(' > Loading model weights.')
nn_L63.model.load_weights('weights/best-weights'+nn_L63.suffix+'.h5')


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


def remove_fp(xt, last_ind=1000) :
    '''
    Replaces fixed point orbits by NaNs. 
    '''
    std_x = np.std(xt[-last_ind:], axis=0)[...,:3]
    std_x = np.sum(std_x, axis=-1)
    fp_indexes = np.where(std_x < 1.)
    fp_theta, fp_orb = fp_indexes[0], fp_indexes[1]
    fill_nans = np.zeros((xt.shape[0], xt.shape[-1]))*np.nan
    if len(fp_theta>0) :
        for i in range(len(fp_theta)) :
            xt[:,fp_theta[i],fp_orb[i]] = fill_nans
    return xt 


mesh_len, n_ic = 20, 25

rhos = np.linspace(26.5, 32., mesh_len)
betas = np.linspace(1.5, 3.2, mesh_len)

rhos_betas = np.array(np.meshgrid(rhos, betas)).T.reshape(-1,2)
rhos_betas = np.swapaxes(np.dstack([rhos_betas]*n_ic),1,2).reshape(-1,2)

sdir = 'dataset/'+datetime.today().strftime('%d%m%Y')+'/'
try : 
    os.mkdir(sdir)
    print(' > Created folder %s.'%sdir)
except : pass

#index_valid = np.random.randint(0, xt_truth.shape[0]-1, n_ic)
#x0 = xt_truth[index_valid]

# New x0 obtained by LHS sampling
min_bounds, max_bounds = np.array([-25.,-25.,0.]), np.array([25.,25.,50.])
delta_bounds = max_bounds-min_bounds
x0 = lhs(3, samples=n_ic)*delta_bounds + min_bounds

spinup = 300
n_steps_valOrb = 20000 + spinup

Thetas, Errors = rhos_betas, np.zeros(mesh_len**2*n_ic)*np.nan
X0 = np.array([x0 for i in range(mesh_len**2)]).reshape(-1,3)
ic = np.zeros((mesh_len**2*n_ic, 5))
ic[:,:3], ic[:,3:] = X0, Thetas

print(' > Computing output')

output = generate_data(nn_L63, x0=ic, n_steps=n_steps_valOrb, dt=0.05, compute_y=False)
xt_pred = output['x']
if n_ic == 1 :
    xt_pred = xt_pred[:,0]

# Removing spinup
xt_pred = xt_pred[spinup:]


xt_pred = np.array([xt_pred[:,n_ic*i:n_ic*(i+1)] for i in range(mesh_len**2)])
xt_pred = np.swapaxes(xt_pred,0,1)
print(' > pred shape : ', xt_pred.shape)
    
np.savez_compressed(sdir+'fixed_sigmas-fhat-pred'+extra_tag+'.npz', xt_pred)

print(' > Done.')
exit()
