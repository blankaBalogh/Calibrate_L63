# Calibration tool for L63 model.

# -------------------------------------------------------------------------------- #
# -------------------------------    Importations   ------------------------------ #
# -------------------------------------------------------------------------------- #
import matplotlib
#matplotlib.use('Agg')

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
import GPy
import matplotlib.pyplot as plt


# -- Parsing aruments
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-a', '--learning_sample', default=1,
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
# xt_truth and yt_truth shape is assumedly (-1,3).
xt_truth = np.load('dataset/xt_truth-50IC.npz')['arr_0']
yt_truth = np.load('dataset/yt_truth-50IC.npz')['arr_0']


# -------------------------------------------------------------------------------- #
# ----------------    2nd EXPERIMENT : LEARNING dx = f(x,theta)   ---------------- #
# -------------------------------------------------------------------------------- #
print('***** 2nd experiment : learning to predict full y_data. ***** ')
print(' ------- Loading learning samples ------- ')

'''
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
    x_data = np.delete(x_data,3,axis=1)
    y_data = np.load('dataset/y_data-a1'+extra_tag+'.npz')['arr_0'][0][...,:3]

if tag=='-amix' :
    print(' > Loading mixed learning sample.')
    x_data = np.load('dataset/x_data-amix'+extra_tag+'.npz')['arr_0']
    y_data = np.load('dataset/y_data-amix'+extra_tag+'.npz')['arr_0'][...,:3]
'''

n_steps_loss, n_snapshots = 200, 50

x0 = np.zeros((n_snapshots, 6))
index_valid = np.random.randint(0, xt_truth.shape[0]-1, n_snapshots)
#x0[...,3:] = np.repeat([10.,28.,8/3], 50).reshape(3, n_snapshots).T

dt = 0.05

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

n_snapshots, n_thetas = 25, 500
n_steps = 20000
sdir = 'dataset/090721/'
try : os.mkdir(sdir)
except : pass


if new_gp :
    print(' > Computing new learning sample for kriging.')
    index_valid = np.random.randint(0, xt_truth.shape[0]-1, n_snapshots)
    x0 = xt_truth[index_valid]

    min_bounds_Th, max_bounds_Th = np.array([10.,26.5,1.5]), np.array([10.,32.,3.2])
    thetas_list = lhs(3, samples=n_thetas)*(max_bounds_Th-min_bounds_Th)+min_bounds_Th
    Thetas = np.array([[theta for i in range(n_snapshots)] for theta in thetas_list])
    Thetas = Thetas.reshape(-1,3)
    #Thetas = np.repeat(thetas_list, n_snapshots).reshape(3,-1).T

    X0 = np.array([x0 for i in range(n_thetas)]).reshape(-1,3)

    ic = np.zeros((n_thetas*n_snapshots,6))
    ic[:,:3], ic[:,3:] = X0, Thetas

    print(' > Computing output')
    
    L63 = Lorenz63()
    output = generate_data(L63, ic, n_steps=n_steps, dt=0.05, compute_y=False)
    xt_pred = output['x']
    xt_pred = np.delete(xt_pred,3,axis=2)
    print('native shape : ', xt_pred.shape)
    #np.savez_compressed(sdir+'raw_xt-obs.npz', xt_pred)

    xt_pred = np.array([[x[i*n_snapshots:(i+1)*n_snapshots] for i in range(n_thetas)] \
        for x in xt_pred])
    print('predicted shape (after reshape) : ', xt_pred.shape)


    # Computing errors
    # Computing predicted orbits standard deviation
    mean_pred = np.mean(xt_pred, axis=(0))[...,:3]
    mean_pred = np.mean(mean_pred, axis=(1))

    std_pred = np.std(xt_pred, axis=(0))[...,:3]
    std_pred = np.mean(std_pred, axis=(1))    # Computing mean values on different x0 IC

    # Computing truth standard deviation
    mean_truth = np.mean(xt_truth, axis=(0))
    std_truth = np.std(xt_truth, axis=(0))

    # Computing error (MSE) on STDs
    err_mean = np.mean((mean_pred-mean_truth)**2, axis=1)
    err_std = np.mean((std_pred-std_truth)**2, axis=1)

    saving_gp = True  
    extra_tag = extra_tag
    if saving_gp :
        print(' > Saving GP learning sample.')
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

thetas_list = thetas_list[:,1:]
## Calibration
err_type='mean+std'
if err_type=='std' :
    y = np.copy(err_std)
else :
    y = err_std+err_mean

mean_thetas, std_thetas = np.mean(thetas_list, axis=0), np.std(thetas_list, axis=0)
mean_y, std_y = np.mean(err_std, axis=0), np.std(err_std, axis=0)

norm_gp = True
if norm_gp :
    thetas_list = (thetas_list-mean_thetas)/std_thetas
    y = (y-mean_y)/std_y

y = err_std.reshape(-1,1)

# Removing sigmas
thetas_list = thetas_list.reshape(-1,2)
kernel = GPy.kern.Matern52(input_dim=2, ARD=True)
gp = GPy.models.GPRegression(thetas_list, y, kernel)
gp.optimize(messages=True)
print(gp)
print(gp.kern.lengthscale)

print('\n -------  Optimization  -------')

norms_gp = np.array([mean_thetas, mean_y, std_thetas, std_y])

loss_kriging = compute_loss_kriging(gp, norm_gp=norm_gp, norms=norms_gp)
    
theta_to_update = [27., 2.5]
print(' > initial theta value : ', theta_to_update)

eps, tol = 1e-1, 1e-3

res = minimize(loss_kriging, theta_to_update, method='BFGS', 
        callback=callbackF, options={'eps':eps}, tol=tol)
    
theta_star = res.x

print(' > optimal theta value : ', theta_star)


### END OF SCRIPT ###
print(' > Done.')
exit()
