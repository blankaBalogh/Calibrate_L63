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
from ML_model_param import ML_model
from data import generate_data
from metrics import *
import gpflow

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus :
    tf.config.experimental.set_memory_growth(gpu, True)

tag         = '-a1'
extra_tag   = '-1d'
exp         = '1d'


# -------------------------------------------------------------------------------- #
# ---------------------    Loading available 'observations'   -------------------- #
# -------------------------------------------------------------------------------- #

# 'Observations' can be loaded from the 'datatset' directory.  
# They are then used to calculate longterm metric (e.g., mean, std, covaX0 = np.array([x0 for i in range(mesh_len**2)]).reshape(-1,3)riance)
xt_truth = np.load('dataset/xt_truth.npz')['arr_0']
yt_truth = np.load('dataset/yt_truth.npz')['arr_0']

# Computing truth standard deviation
mean_truth = np.mean(xt_truth, axis=0)
std_truth = np.std(xt_truth, axis=0)


# -------------------------------------------------------------------------------- #
# -----------------------------    Loading NN models   --------------------------- #
# -------------------------------------------------------------------------------- #

# Loading lhs learning sample
print(' > Loading learning sample of LHS sample.')
x_data = np.load('dataset/x_data-a1'+extra_tag+'.npz')['arr_0'][0]
y_data = np.load('dataset/y_data-a1'+extra_tag+'.npz')['arr_0'][0][...,:3]
x_data = np.delete(x_data, 3, axis=-1)

x_data = np.delete(x_data,3,axis=-1)
y_data = y_data[:,-1]
y_data = y_data.reshape(-1,1)


# Learning fhat_betas
print('\n ------ Learning fhat_thetas ------- ')

# Normalization of x & y data
mean_x, std_x = np.mean(x_data, axis=0), np.std(x_data, axis=0)
mean_y, std_y = np.mean(y_data, axis=0), np.std(y_data, axis=0)
x_data = (x_data-mean_x)/std_x
y_data = (y_data-mean_y)/std_y

if exp=='2d' :
    layers = [1024, 512, 256, 128, 64, 32, 16]
else :
    layers = [256, 128, 64, 32, 16]



dic_NN = {'name':'f_orb', 'in_dim':x_data.shape[1], 'out_dim':y_data.shape[1], 
        'nlays':layers, 'dropout':False}
dic_NN_stand = {'name':'f_orb', 'in_dim':3, 'out_dim':1, 
        'nlays':layers, 'dropout':False}
nn_L63, nn_L63_stand = ML_model(dic_NN), ML_model(dic_NN_stand)
nn_L63.norms = [mean_x, mean_y, std_x, std_y]
nn_L63_stand.norms = [mean_x[:3], mean_y, std_x[:3], std_y]
nn_L63.suffix, nn_L63_stand.suffix = tag+extra_tag, tag+'-1d-std'
print(nn_L63.model.summary())

# Loading best model weights
print(' > Loading model weights.')
#nn_L63.model.load_weights('weights/best-weights'+nn_L63.suffix+'.h5')
nn_L63.model.load_weights('weights/best-weights'+nn_L63.suffix+'.h5')
nn_L63_stand.model.load_weights('weights/best-weights'+nn_L63_stand.suffix+'.h5')


"""
# -------------------------------------------------------------------------------- #
# ------------------------    Generating 'observations'   ------------------------ #
# -------------------------------------------------------------------------------- #
# The purpose of this section is to generate 'observations', i.e. a long orbit from 
# a given initial condition. 

print(" > Computing 'observations'.")

ic      = np.zeros(6)
ic[:3]  = x_data[0,:3]      # initial condition
ic[3:]  = np.array([10.,28.,x_data[0,-1]])

n_steps = x_data.shape[0]   # orbit length in timesteps


L63     = Lorenz63()        # plain L63 model
output  = generate_data(L63, x0=ic, n_steps=n_steps, dt=0.05, compute_y=False)
xt_pred = output['x']
"""


# -------------------------------------------------------------------------------- #
# -----------------------------    Computing errors    --------------------------- #
# -------------------------------------------------------------------------------- #
print(' ------- Generating validation samples ------- ')
dt          = 0.05
n_val_eval  = 60000
x0_test     = xt_truth[0]
n_val       = 2000
n_snapshots = 1
count       = 0
alpha       = 0.5

len_linspace    = 10
biased_sigmas   = np.linspace(9.5, 10.5, len_linspace)
biased_rhos     = np.linspace(26.5, 29.5, len_linspace)
biased_thetas   = np.transpose([np.tile(biased_sigmas, len_linspace), 
    np.repeat(biased_rhos, len_linspace)])

results = np.zeros((100, 5))*np.nan
results[...,:2] = biased_thetas
#results[0, 0] = 'biased sigma'
#results[0, 1] = 'biased rho'
#results[0, 2] = 'optimal beta'
#results[0, 3] = 'err optimal beta'
#results[0, 4] = 'err standard beta'



def compute_loss_data(xt_pred) :
    '''
    Loss function.
    '''
    std_NN      = np.std(xt_pred, axis=(0,1))
    err_std     = np.mean((std_NN-std_truth)**2)
    mean_NN     = np.mean(xt_pred, axis=(0,1))
    err_mean    = np.mean((std_NN-std_truth)**2)
    err = alpha*err_std + (1.-alpha)*err_mean
    print('error : ', err)
    return err


def loss_kriging(beta_to_update) :
    '''
    Loss function to optimize beta value after kriging. 
    '''
    beta_ = np.array([beta_to_update])
    print('Optimal beta value : %f.' % beta_)
    err = gp.predict(beta_) 
    return err[0,0]


for (biased_sigma, biased_rho) in biased_thetas :
    print('\n Op. no. %d/100'%(count+1))
    print(' > biased sigma value    : ', biased_sigma)
    print(' > biased rho value      : ', biased_rho)
    beta_to_update = 2.5

    # Optimizing input parameter values
    n_train_kriging = 50
    betas_list = np.linspace(2., 4., n_train_kriging)
    ic = np.zeros((n_train_kriging,6))*np.nan
    ic[...,3] = np.repeat(biased_sigma, n_train_kriging)
    ic[...,4] = np.repeat(biased_rho, n_train_kriging)
    ic[...,5] = betas_list
    ic[...,:3] = np.repeat(xt_truth[0], n_train_kriging).reshape(3,-1).T

    output = generate_data(nn_L63, x0=ic, n_steps=n_val_eval, dt=0.05, compute_y=False)
    xt_pred = output['x'][...,:3]
    errors = compute_loss_data(xt_pred)
   
    # kriging
    betas   = betas_list.reshape(-1,1)
    k       = gpflow.kernels.Matern52(variance=1., lengthscales=1.)

    normalize_data_kriging = True
    mean_betas, std_betas   = np.mean(betas), np.std(betas)
    mean_err, std_err       = np.mean(errors), np.std(errors)
    if normalize_data_kriging :
        betas_list  = (betas_list-mean_betas)/std_betas
        errors      = (errors-mean_err)/std_err
        beta_to_update = (beta_to_update-mean_betas)/std_betas

    m       = gpflow.models.GPR(data=(betas_list, errors), kernel=k)#, mean_function=None)
    opt     = gpflow.optimizers.Scipy()
    opt_logs= opt.minimize(m.training_loss, m.trainable_variables)

    res = minimize(loss_kriging, beta_to_update, method='BFGS', tol=1e-2, 
            options={'eps':1e-3})
    beta_opt            = res.x[0]
    if normalize_data_kriging :
        beta_opt = beta_opt*std_betas+mean+betas
    results[count, 2]   = beta_opt

    betas_optim = np.repeat(beta_opt, 1)
    betas_optim = betas_optim.reshape(1,1)

    ic_test = np.zeros(6)
    ic_test[:3] = x0_test
    ic_test[4], ic_test[5] = biased_sigma, biased_rho
    ic_test[5] = betas_optim[0]
    xt_fhat_standard    = generate_data(nn_L63_stand, x0_test, n_steps=n_val_eval, 
            dt=dt, compute_y=False)['x']
    xt_fhat_optim       = generate_data(nn_L63, x0_test, betas=betas_optim, 
            n_steps=n_val_eval, dt=dt, compute_y=False)['x']

    print(' ------- Computing errors ------- ')
    std_stand = np.std(xt_fhat_standard[:,0], axis=0)
    std_optim = np.std(xt_fhat_optim[:,0], axis=0)
    std_truth = np.std(xt_truth, axis=0)
    
    mean_stand = np.mean(xt_fhat_standard[:,0], axis=0)
    mean_optim = np.mean(xt_fhat_optim[:,0], axis=0)
    mean_truth = np.mean(xt_truth, axis=0)

    err_std_stand = np.mean((std_stand-std_truth)**2)
    err_std_optim = np.mean((std_optim-std_truth)**2)
    err_mean_stand = np.mean((mean_stand-mean_truth)**2)
    err_mean_optim = np.mean((mean_optim-mean_truth)**2)

    err_stand = alpha*err_std_stand + (1.-alpha)*err_mean_stand
    err_optim = alpha*err_std_optim + (1.-alpha)*err_mean_optim

    results[count, 3] = err_optim
    results[count, 4] = err_stand

    print(' ------- RESULTS ------- ')
    print(' > Error standard beta value : ', err_stand)
    print(' > Error optimal beta value  : ', err_optim)
    print(' > Optimal beta value : ', beta_opt)    

    count += 1

np.savez_compressed('results-linspace-STDOnly-newNN-kriging-1ts.npz', results)
#np.savez_compressed('results-linspace-meanSTD.npz', results)

#print(' > Saving results to dir results/')
#print('   Generating xt_fhat_standard...')
#np.savez_compressed('results/xt_fhat_standard-190321.npz', xt_fhat_standard)
#print('   Generating xt_fhat_optim...')
#np.savez_compressed('results/xt_fhat_optim-190321.npz', xt_fhat_optim)
#print('\n > Datasets has been successfully generated.')




