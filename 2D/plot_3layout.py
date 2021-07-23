import matplotlib
matplotlib.use('Agg')

import numpy as np
import GPy
import matplotlib.pyplot as plt
from L63_mix import Lorenz63
from data import generate_data
from mpl_toolkits import mplot3d
from pyDOE import lhs
from datetime import datetime

from scipy.optimize import minimize
from metrics import compute_loss_kriging


def remove_fp(xt, last_ind=5000) :
    '''
    Replaces fixed point orbits by NaNs. 
    '''
    std_x = np.std(xt[-last_ind:], axis=0)[...,:3]
    std_x = np.sum(std_x, axis=-1)
    fp_indexes = np.where(std_x < 6.)
    fp_theta, fp_orb = fp_indexes[0], fp_indexes[1]
    fill_nans = np.zeros((xt.shape[0], xt.shape[-1]))
    if len(fp_theta>0) :
        for i in range(len(fp_theta)) :
            xt[:,fp_theta[i],fp_orb[i]] = fill_nans
    return xt


# Defining directories, etc.
today   = datetime.today().strftime('%d%m%Y')
fdir    = 'dataset/'        # root dir
param   = 'sigmas'          # fixed parameter : sigma=10.
alpha   = 0.5               # see below : err_obs
datadir = fdir+today+'/'    # output directory
tag     = '-a1'             # model to load
et      = '-larger-largerNN'# model to load
log_errors = True           # log errors for final plot

# Genrating and saving f-data ? 
gen_data    = True     
save_data   = False


# Loading truth dataset
xt_truth = np.load(fdir+'xt_truth.npz')['arr_0']
std_truth = np.std(xt_truth, axis=0)
mean_truth = np.mean(xt_truth, axis=0)


# -------------------------------------------------------------------------------- #
# ---------------------------------    Kriging   --------------------------------- #
# -------------------------------------------------------------------------------- #
print(' > Kriging.')
# Loading training dataset for kriging metamodel
train_thetas = np.load(datadir+'train_thetas_gp-fhat-a1'+et+'.npz')['arr_0']
train_errors_mean = np.load(datadir+'train_errors_mean_gp-fhat-a1'+et+'.npz')['arr_0']
train_errors_std = np.load(datadir+'train_errors_std_gp-fhat-a1'+et+'.npz')['arr_0']

err_obs = (1-alpha)*train_errors_mean + alpha*train_errors_std
err_obs = err_obs.reshape(-1,1)


# Fitting kriging metamodel
kernel = GPy.kern.Matern52(input_dim=2, ARD=True)
m = GPy.models.GPRegression(train_thetas, err_obs, kernel)
m.optimize(messages=True)
print(m)
print(m.kern.lengthscale)


# Predicting Errors -- final plot
mesh_len = 20       # number of rho or beta values & number of orbits

x, y = np.linspace(26.,32.,mesh_len), np.linspace(1.5,3.2,mesh_len)
sigmas_rhos = np.array(np.meshgrid(x, y)).T.reshape(-1,2)
Thetas_to_predict = np.copy(sigmas_rhos)

pred_errors = m.predict(Thetas_to_predict)[0][:,0]



# -------------------------------------------------------------------------------- #
# ---------------------------------    NN model   -------------------------------- #
# -------------------------------------------------------------------------------- #
print(' > NN.')

# NN-predicted errors
print('   Loading NN-predicted errors')
xt_NN   = np.load(datadir+'fixed_sigmas-fhat-pred'+et+'.npz')['arr_0']

# Saving & formatting ic for f model (see below)
x0      = xt_NN[0]
shape   = list(x0.shape)
shape[-1] = 6
shape   = tuple(shape)
ic      = np.zeros(shape)
ic[...,:3], ic[...,4:] = x0[...,:3], x0[...,3:]
sigmas = np.repeat(10.,np.product(shape[:-1])).reshape(shape[:-1])
ic[...,3] = sigmas

# Computing errors
xt_NN_nans = remove_fp(xt_NN)   # removing fixed points

std_NN = np.nanstd(xt_NN_nans[...,:3], axis=0)
std_NN[std_NN==0.] = np.nan

mean_NN = np.nanmean(xt_NN_nans[...,:3], axis=0)
mean_NN[mean_NN==0.] = np.nan

if len(std_NN.shape) == 3 :
    std_NN = np.nanmean(std_NN, axis=1)
    mean_NN = np.nanmean(mean_NN, axis=1)

err_std_NN = np.mean((std_NN-std_truth)**2, axis=1)
err_mean_NN = np.mean((mean_NN[:,:3]-mean_truth)**2, axis=1)

err_NN = alpha*err_std_NN + (1-alpha)*err_mean_NN



# -------------------------------------------------------------------------------- #
# ---------------------------------    L63 model   ------------------------------- #
# -------------------------------------------------------------------------------- #
print(' > L63.')
print('   Computing errors with truth function f')
n_steps_val = xt_NN.shape[0]
L63         = Lorenz63()
output      = generate_data(L63, ic, n_steps=n_steps_val, dt=0.05, compute_y=False)
xt_L63      = output['x']

std_L63 = np.std(xt_L63[...,:3], axis=0)
mean_L63 = np.mean(xt_L63[...,:3], axis=0)

if len(std_L63.shape) == 3 :
    std_L63 = np.mean(std_L63, axis=1)
    mean_L63 = np.mean(mean_L63, axis=1)

err_std_L63 = np.mean((std_L63-std_truth)**2, axis=1)
err_mean_L63 = np.mean((mean_L63[:,:3]-mean_truth)**2, axis=1)

err_L63 = alpha*err_std_L63 + (1-alpha)*err_mean_L63



# -------------------------------------------------------------------------------- #
# ----------------------------------    Layout   --------------------------------- #
# -------------------------------------------------------------------------------- #
print(' > Plotting layout & optimization.')
rhos    = Thetas_to_predict[:,0].reshape(mesh_len,mesh_len)
betas   = Thetas_to_predict[:,1].reshape(mesh_len,mesh_len)

if log_errors :
    err_NN  = np.log(err_NN)
    err_L63 = np.log(err_L63)
    min_pred= np.min(pred_errors)

    if min_pred<0 :
        eps = 1e-6
        min_pred = -min_pred+eps

    pred_errors = np.log(pred_errors+min_pred)

errors_NN   = err_NN.reshape(mesh_len,mesh_len,1)
errors_L63  = err_L63.reshape(mesh_len,mesh_len,1)
errors_pred = pred_errors.reshape(mesh_len,mesh_len,1)

x, y        = rhos, betas
plot_legend = 'sigma=10'
xlabel, ylabel  = 'rho', 'beta'
truth_x, truth_y= 28., 8/3


# Optimization
print('   --> Optimization.')
n_iter = 0

def callbackF(x_) :
    '''
    Function to print all intermediate values
    '''
    global n_iter
    print("Iteration no.", n_iter)
    print("theta value : ", x_)
    n_iter += 1

loss_kriging    = compute_loss_kriging(m, norm_gp=False)
theta_to_update = [27., 2.4]        # optimization ic
bounds = ((26.5,32.),(1.5,3.2))     # bounds (if 'L-BFGS-B' optimizer is used)

print('   initial theta value : ', theta_to_update)
eps, tol = 1e-1, 1e-3
res = minimize(loss_kriging, theta_to_update, method='BFGS', callback=callbackF, 
        options={'eps':eps})
theta_star = res.x
print('   optimal theta value : ', theta_star,'.\n')



# Plotting layouts
print('   --> Plotting layout.')
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,7))

im=ax[0].pcolormesh(x,y,errors_L63[:,:,0], shading='auto', vmin=-3., vmax=1.5)
ax[0].axvline(truth_x, color='red', ls='--')
ax[0].axhline(truth_y, color='red', ls='--')
ax[0].set_xlabel(xlabel)
ax[0].set_ylabel(ylabel)
ax[0].set_title('m (obtained with $f$)')

im=ax[1].pcolormesh(x,y,errors_NN[:,:,0], shading='auto', vmin=-3., vmax=1.5)
ax[1].axvline(truth_x, color='red', ls='--')
ax[1].axhline(truth_y, color='red', ls='--')
ax[1].set_xlabel(xlabel)
ax[1].set_ylabel(ylabel)
ax[1].set_title('$\hat{m}$ (obtained with $\hat{f}$)')
#fig.colorbar(im, ax=ax[1])

im=ax[2].pcolormesh(x,y,errors_pred[:,:,0], shading='auto', vmin=-3., vmax=1.5)
ax[2].axvline(truth_x, color='red', ls='--')
ax[2].axhline(truth_y, color='red', ls='--')
ax[2].set_xlabel(xlabel)
ax[2].set_ylabel(ylabel)
ax[2].set_title('kriged m (kriging on $\hat{f}$)')
fig.colorbar(im, ax=ax[2])

plt.tight_layout()
plt.savefig('figures/layout-fhat-fhatkrig-'+today+et+'-noNans.png')



