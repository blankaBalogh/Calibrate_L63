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
from data import generate_data, generate_LHS, generate_data_solvers, generate_x0 
from metrics import *

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
#    print('> New GP learning sample.')xt_fixed_betas = np.load(fdir+'090621/fixed_'+param+'-fhat-pred-090621.npz')['arr_0']



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


mesh_len, n_ic = 10, 50

rhos = np.linspace(26.5, 32., mesh_len)
betas = np.linspace(1.5, 3.2, mesh_len)

rhos_betas = np.array(np.meshgrid(rhos, betas)).T.reshape(-1,2)
rhos_betas = np.swapaxes(np.dstack([rhos_betas]*n_ic),1,2).reshape(-1,2)

sdir = 'dataset/'
try : 
    os.mkdir(sdir)
    print(' > Created folder %s.'%sdir)
except : pass

index_valid = np.random.randint(0, xt_truth.shape[0]-1, n_ic)
x0 = xt_truth[index_valid]

n_steps_valOrb = 20000
   
Thetas, Errors = np.zeros((mesh_len**2*n_ic, 3))*np.nan, np.zeros(mesh_len**2*n_ic)*np.nan
Thetas[:,[1,2]] = rhos_betas
Thetas[:,0] = np.repeat(10., mesh_len**2*n_ic)
    
X0 = np.array([x0 for i in range(mesh_len**2)]).reshape(-1,3)

ic = np.zeros((mesh_len**2*n_ic, 6))
ic[:,:3], ic[:,3:] = X0, Thetas

print(' > Computing output')
    
L63 = Lorenz63()
output = generate_data(L63, ic, n_steps=n_steps_valOrb, dt=0.05, compute_y=False)
xt_pred = output['x']
    
np.savez_compressed(sdir+'fixed_sigmas-fhat-pred-070721.npz', xt_pred)

print(' > Done.')
exit()
