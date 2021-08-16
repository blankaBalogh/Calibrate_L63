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
# -----------------------------    Generating data   ----------------------------- #
# -------------------------------------------------------------------------------- #


mesh_len    = 20    # Number of values for a given parameter. Mesh size = mesh_len**2.
n_ic        = 25    # Number of x initial conditions

if exp=='2d' :
    rhos    = np.linspace(26., 32., mesh_len)      # list of rho values to predict
    betas   = np.linspace(1.5, 3.2, mesh_len)
    rhos_betas = np.array(np.meshgrid(rhos, betas)).T.reshape(-1,2)
else :
    rhos    = np.repeat(28., mesh_len**2).reshape(-1,1)
    betas   = np.linspace(1.5,3.2,mesh_len**2).reshape(-1,1)
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

L63 = Lorenz63()
output = generate_data(L63, x0=ic, n_steps=n_steps_valOrb, dt=0.05, compute_y=False)
xt_pred = output['x']
if n_ic == 1 :
    xt_pred = xt_pred[:,0]

# Removing spinup
xt_pred = xt_pred[spinup:]

# Reshaping resulting array
xt_pred = np.array([xt_pred[:,n_ic*i:n_ic*(i+1)] for i in range(mesh_len**ndim)])
xt_pred = np.swapaxes(xt_pred,0,1)
print(' > pred shape : ', xt_pred.shape)
    
np.savez_compressed(sdir+'fixed_sigmas-f-pred'+extra_tag+'.npz', xt_pred)

print(' > Done.')
exit()
