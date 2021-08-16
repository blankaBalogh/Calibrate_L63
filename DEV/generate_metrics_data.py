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
parser.add_argument('-a', '--learning_sample', default=1,
        help='Learning sample selection : orbit (=2) or lhs (=1).')
parser.add_argument('-et', '--extra_tag', type=str, default='', 
        help='Adds an extra tag. Useful to save new datasets.')
parser.add_argument('-exp', '--experience', type=str, default='2d',
        help="Experience type : '2d' or '1d'.")

args    = parser.parse_args()

tag = '-a'+str(args.learning_sample)
extra_tag = args.extra_tag
exp = args.experience
if exp=='1d' :
    extra_tag += '-1d'
    ndim = 1
else :
    ndim = 2

if tag=='-a1' : learning_sample = 'LHS'
else : learning_sample = 'orbits'

print()
print('> Learning sample : %s.'%learning_sample)

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


# Loading 'orbits' learning sample
if tag=='-a2' :
    print(' > Loading learning sample of orbits.')
    x_data = np.load('dataset/x_data-a2'+extra_tag+'.npz')['arr_0']
    y_data = np.load('dataset/y_data-a2'+extra_tag+'.npz')['arr_0'][...,:3]
    x_data, y_data = np.swapaxes(x_data,0,1), np.swapaxes(y_data,0,1) 
    x_data = x_data.reshape(-1, x_data.shape[-1])
    y_data = y_data.reshape(-1, y_data.shape[-1])
    if exp=='1d' :
        x_data = np.delete(x_data,3,axis=-1)
        y_data = y_data[:,-1]
        y_data = y_data.reshape(-1,1)


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

if exp=='2d' : 
    liste_dims = [0,1,2,4,5]
else :
    liste_dims = [0,1,2,5]
    y_data = y_data[:,-1]
    y_data = y_data.reshape(-1,1)
    mean_y, std_y = mean_y[-1], std_y[-1]

x_data = x_data[...,liste_dims]
mean_x, std_x = mean_x[liste_dims], std_x[liste_dims]
print('mean_x : ', mean_x)

# Setting up NN model
if exp=='2d' :
    layers = [1024, 512, 256, 128, 64, 32, 16]
else :
    layers = [256, 128, 64, 32, 16]



dic_NN = {'name':'f_orb', 'in_dim':x_data.shape[1], 'out_dim':y_data.shape[1], 
        'nlays':layers, 'dropout':False, 'exp':exp}
nn_L63 = ML_model(dic_NN)
nn_L63.norms = [mean_x, mean_y, std_x, std_y]
nn_L63.suffix = tag+extra_tag
print(nn_L63.model.summary())

# Loading best model weights
print(' > Loading model weights.')
nn_L63.model.load_weights('weights/best-weights'+nn_L63.suffix+'.h5')



# -------------------------------------------------------------------------------- #
# -----------------------------    Generating data   ----------------------------- #
# -------------------------------------------------------------------------------- #

mesh_len    = 20    # Number of values for a given parameter. Mesh size = mesh_len**2.
n_ic        = 25    # Number of x initial conditions

if exp=='2d' :
    rhos    = np.linspace(26., 32., mesh_len)      # list of rho values to predict
    betas   = np.linspace(1.5, 3., mesh_len)
    rhos_betas = np.array(np.meshgrid(rhos, betas)).T.reshape(-1,2)
else :
    rhos    = np.repeat(28., mesh_len**2).reshape(-1,1)
    betas   = np.linspace(1.5,3.,mesh_len**2).reshape(-1,1)
    rhos_betas = np.concatenate((rhos, betas), axis=-1)

rhos_betas = np.swapaxes(np.dstack([rhos_betas]*n_ic),1,2).reshape(-1,2)
sigmas = np.repeat(10., rhos_betas.shape[0]).reshape(-1,1)
Thetas = np.concatenate((sigmas, rhos_betas), axis=-1)

sdir = 'dataset/'+datetime.today().strftime('%d%m%Y')+'/'
try : 
    os.mkdir(sdir)
    print(' > Created folder %s.'%sdir)
except :
    pass

# New x0 obtained by LHS sampling
min_bounds, max_bounds = np.array([-25.,-25.,0.]), np.array([25.,25.,50.])
delta_bounds = max_bounds-min_bounds
x0 = lhs(3, samples=n_ic)*delta_bounds + min_bounds

spinup = 300
n_steps_valOrb = 20000 + spinup

Errors  = np.zeros(mesh_len**ndim*n_ic)*np.nan
X0      = np.array([x0 for i in range(mesh_len**ndim)]).reshape(-1,3)
ic      = np.zeros((mesh_len**ndim*n_ic, 6))
ic[:,:3], ic[:,3:] = X0, Thetas

print(' > Computing output')


output = generate_data(nn_L63, x0=ic, n_steps=n_steps_valOrb, dt=0.05, compute_y=False)
L63 = Lorenz63()
output_L63 = generate_data(L63, x0=ic, n_steps=n_steps_valOrb, dt=0.05, compute_y=False)
xt_pred = output['x']
xt_L63 = output_L63['x']

if n_ic == 1 :
    xt_pred = xt_pred[:,0]
    xt_L63 = xt_L63[:,0]

# Removing spinup
xt_pred = xt_pred[spinup:]
xt_L63 = xt_L63[spinup:]

# Reshaping resulting array
xt_pred = np.array([xt_pred[:,n_ic*i:n_ic*(i+1)] for i in range(mesh_len**ndim)])
xt_L63 = np.array([xt_L63[:,n_ic*i:n_ic*(i+1)] for i in range(mesh_len**ndim)])
xt_pred = np.swapaxes(xt_pred,0,1)
xt_L63 = np.swapaxes(xt_L63,0,1)
print(' > pred shape : ', xt_pred.shape)
    
np.savez_compressed(sdir+'fixed_sigmas-fhat-pred'+extra_tag+'.npz', xt_pred)
np.savez_compressed(sdir+'fixed_sigmas-f-pred'+extra_tag+'.npz', xt_L63)

print(' > Done.')
exit()
