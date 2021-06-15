# Plots layout of aggregated metric computed with f and the same field after kriging. 
# 'param' specifies the parameter with a fixed value. 

import numpy as np
import GPy
import matplotlib.pyplot as plt
from L63_mix import Lorenz63
from data import generate_data
from mpl_toolkits import mplot3d
from pyDOE import lhs

param = 'betas'
log_errors = True

gen_data = True
save_data = False


# Loading data
fdir = 'dataset/'
#fdir = '/cnrm/amacs/USERS/baloghb/calibration_L63/v6/new_exp2/dataset/'

#xt_fixed_betas = np.load(fdir+'090621/fixed_'+param+'-fhat-pred-090621.npz')['arr_0']
#x0 = xt_fixed_betas[0]
x0 = np.load(fdir+'090621/fixed_'+param+'-x0-090621.npz')['arr_0']
#xt_fixed_betas = np.array([[x[i*50:(i+1)*50] for i in range(100)] for x in xt_fixed_betas])
# Resulting array of shape : (20000,100,50,6).

min_bounds = np.array([-35., -35., 0., 7., 26.5, 1.5])
max_bounds = np.array([35., 35., 60., 13., 32., 3.2])

np.random.seed(100)
x0 = lhs(6, samples=500)*(max_bounds-min_bounds) + min_bounds


# Observations

if gen_data :
    L63 = Lorenz63()
    output = generate_data(L63, x0=x0, n_steps=20000, dt=0.05, compute_y=False)
    xt_obs = output['x']
    if save_data :
        np.savez_compressed(fdir+'090621/train_x-f.npz', xt_obs)
else :
    xt_obs = np.load(fdir+'090621/train_x-f.npz')['arr_0']

train_thetas = xt_obs[0,:,3:]


# Truth
#xt_truth = np.load(fdir+'xt_truth.npz')['arr_0']
xt_truth = np.load(fdir+'x_data-a2-newTruth.npz')['arr_0']#[:,:,:3]

# Computing standard deviations
std_truth = np.mean(np.std(xt_truth[:,:,:3], axis=0), axis=0)
mean_truth = np.mean(np.mean(xt_truth[:,:,:3], axis=0), axis=0)

std_obs = np.std(xt_obs[:,:,:3], axis=0)
mean_obs = np.mean(xt_obs, axis=0)

err_mean_obs = np.mean((mean_obs[:,:3]-mean_truth)**2, axis=1)
err_std_obs = np.mean((std_obs-std_truth)**2, axis=1)
alpha = 0.5
err_obs = alpha*err_std_obs + (1-alpha)*err_mean_obs
err_obs = err_obs.reshape(-1,1)

if save_data :
    np.savez_compressed(fdir+'090621/gp_train_thetas.npz', train_thetas)
    np.savez_compressed(fdir+'090621/gp_train_errors.npz', err_obs)


# Fitting GP model
kernel = GPy.kern.Matern52(input_dim=3, ARD=True)

m = GPy.models.GPRegression(train_thetas, err_obs, kernel)
m.optimize(messages=True)
print(m)
print(m.kern.lengthscale)

# Predicting Errors
if param=='rhos' :
    x, y = np.linspace(7.,13.,10), np.linspace(1.5,3.2,10)
    sigmas_rhos = np.array(np.meshgrid(x, y)).T.reshape(-1,2)
    z = np.repeat(28., 100)
    Thetas_to_predict = np.zeros((100,3))
    Thetas_to_predict[:,0] = sigmas_rhos[:,0]
    Thetas_to_predict[:,1] = z
    Thetas_to_predict[:,2] = sigmas_rhos[:,1]

elif param=='betas' :
    x, y = np.linspace(7.,13.,10), np.linspace(26.5,32.,10) 
    sigmas_rhos = np.array(np.meshgrid(x, y)).T.reshape(-1,2)
    z = np.repeat(8/3, 100)
    Thetas_to_predict = np.zeros((100,3))
    Thetas_to_predict[:,0] = sigmas_rhos[:,0]
    Thetas_to_predict[:,1] = sigmas_rhos[:,1]
    Thetas_to_predict[:,2] = z

elif param=='sigmas' :
    x, y = np.linspace(26.5,32.,10), np.linspace(1.5,3.2,10)
    sigmas_rhos = np.array(np.meshgrid(x, y)).T.reshape(-1,2)
    z = np.repeat(10., 100)
    Thetas_to_predict = np.zeros((100,3))
    Thetas_to_predict[:,0] = z
    Thetas_to_predict[:,1] = sigmas_rhos[:,0]
    Thetas_to_predict[:,2] = sigmas_rhos[:,1]

# Validation + data to plot
pred_errors = m.predict(Thetas_to_predict)[0][:,0]

# Loading validation data 
xt_obs = np.load(fdir+'x_data-f-'+param+'-valid-LB.npz')['arr_0']
#xt_obs = np.array([[x[i*50:(i+1)*50] for i in range(100)] for x in xt_obs])

std_obs = np.std(xt_obs[:,:,:3], axis=0)
mean_obs = np.mean(xt_obs[:,:,:3], axis=0)

err_std_obs = np.mean((std_obs-std_truth)**2, axis=1)
err_mean_obs = np.mean((mean_obs[:,:3]-mean_truth)**2, axis=1)

err_obs = alpha*err_std_obs + (1-alpha)*err_mean_obs


sigmas = Thetas_to_predict[:,0].reshape(10,10)
rhos = Thetas_to_predict[:,1].reshape(10,10)
betas = Thetas_to_predict[:,2].reshape(10,10)

if log_errors :
    err_obs = np.log(err_obs)
    pred_errors = np.log(pred_errors)

errors_obs = err_obs.reshape(10,10,1)
errors_pred = pred_errors.reshape(10,10,1)

#np.savez_compressed(fdir+'090621/errors_gp_predicted.npz', errors_pred)


if param=='rhos' :
    print(' > fixed param : rho.')
    x, y = sigmas, betas
    plot_legend = 'rho=28'
    xlabel, ylabel = 'sigma', 'beta'
    truth_x, truth_y = 10., 8/3

elif param=='betas' :
    print(' > fixed param : beta.')
    x, y = sigmas, rhos
    plot_legend = 'beta=8/3'
    xlabel, ylabel = 'sigma', 'rho'
    truth_x, truth_y = 10., 28.

elif param=='sigmas' :
    print(' > fixed param : sigma.')
    x, y = rhos, betas
    plot_legend = 'sigma=10'
    xlabel, ylabel = 'rho', 'beta'
    truth_x, truth_y = 28., 8/3



# Plotting layouts
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,7))

im=ax[0].pcolormesh(x,y,errors_obs[:,:,0], shading='auto', vmin=-5., vmax=0.)
ax[0].axvline(truth_x, color='red', ls='--')
ax[0].axhline(truth_y, color='red', ls='--')
ax[0].set_xlabel(xlabel)
ax[0].set_ylabel(ylabel)
ax[0].set_title('obs err')
fig.colorbar(im, ax=ax[0])

im=ax[1].pcolormesh(x,y,errors_pred[:,:,0], shading='auto', vmin=-5., vmax=0.)
ax[1].axvline(truth_x, color='red', ls='--')
ax[1].axhline(truth_y, color='red', ls='--')
ax[1].set_xlabel(xlabel)
ax[1].set_ylabel(ylabel)
ax[1].set_title('pred err')
fig.colorbar(im, ax=ax[1])

plt.suptitle('errors with '+plot_legend+', observations.')
plt.tight_layout()
plt.show()

