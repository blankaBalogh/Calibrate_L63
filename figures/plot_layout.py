# Plots layouts of : i) mean field & metric per component, ii) aggregated metric over mean fields, 
# iii) std field & metric per component, iv) aggregated metric over std field. 

import numpy as np
import matplotlib.pyplot as plt
from L63_mix import Lorenz63
from data import generate_data
from mpl_toolkits import mplot3d

param = 'sigmas'
log_errors = True

gen_data = True
save_data = True


# Loading data
fdir = '../dataset/'
#fdir = '/cnrm/amacs/USERS/baloghb/calibration_L63/v6/new_exp2/dataset/'

xt_fixed_betas = np.load(fdir+'090621/fixed_'+param+'-fhat-pred-090621.npz')['arr_0']
x0 = xt_fixed_betas[0]
xt_fixed_betas = np.array([[x[i*50:(i+1)*50] for i in range(100)] for x in xt_fixed_betas])
# Resulting array of shape : (20000,100,50,6).

# Observations

if gen_data :
    L63 = Lorenz63()
    output = generate_data(L63, x0=x0, n_steps=20000, dt=0.05, compute_y=False)
    xt_obs = output['x']
    if save_data :
        np.savez_compressed(fdir+'090621/fixed_'+param+'-f-obs-090621-larger.npz', xt_obs)
else :
    xt_obs = np.load(fdir+'090621/fixed_'+param+'-f-obs-090621-larger.npz')['arr_0']

xt_obs = np.array([[x[i*50:(i+1)*50] for i in range(100)] for x in xt_obs])


# Truth
#xt_truth = np.load(fdir+'xt_truth.npz')['arr_0']
xt_truth = np.load(fdir+'x_data-a2-newTruth.npz')['arr_0']#[:,:,:3]


# Computing standard deviations
std_truth = np.mean(np.std(xt_truth[:,:,:3], axis=0), axis=0)
mean_truth = np.mean(np.mean(xt_truth[:,:,:3], axis=0), axis=0)

std_pred = np.std(xt_fixed_betas[:,:,:,:3], axis=0)
std_pred = np.mean(std_pred, axis=1)
mean_pred = np.mean(np.mean(xt_fixed_betas[:,:,:,:3], axis=0), axis=1)

std_obs = np.std(xt_obs[:,:,:,:3], axis=0)
std_obs = np.mean(std_obs, axis=1)
mean_obs = np.mean(np.mean(xt_obs, axis=0), axis=1)

err_mean_obs = (mean_obs[:,:3]-mean_truth)**2
err_std_obs = (std_obs-std_truth)**2

sigmas = mean_obs[:,3].reshape(10,10)
rhos = mean_obs[:,4].reshape(10,10)
betas = mean_obs[:,5].reshape(10,10)


if log_errors :
    err_mean_obs = np.log(err_mean_obs)
    err_std_obs = np.log(err_std_obs)

errors_mean = err_mean_obs.reshape(10,10,3)
errors_std = err_std_obs.reshape(10,10,3)
means = mean_obs[:,:3].reshape(10,10,3)
stds = std_obs.reshape(10,10,3)

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
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10,7))

# Plotting mean values & errors
for i in range(3) :
    im=ax[0,i].pcolormesh(x,y,means[:,:,i], shading='auto')#, vmin=-6., vmax=4.)
    ax[0,i].axvline(truth_x, color='red', ls='--')
    ax[0,i].axhline(truth_y, color='red', ls='--')
    ax[0,i].set_xlabel('sigma')
    ax[0,i].set_ylabel('beta')
    ax[0,i].set_title('$x_{%i}$'%(i+1))
    fig.colorbar(im, ax=ax[0,i])

    im=ax[1,i].pcolormesh(x,y,errors_mean[:,:,i], shading='auto', vmin=-6., vmax=4.)
    ax[1,i].axvline(truth_x, color='red', ls='--')
    ax[1,i].axhline(truth_y, color='red', ls='--')
    ax[1,i].set_xlabel('sigma')
    ax[1,i].set_ylabel('beta')
    fig.colorbar(im, ax=ax[1,i])

plt.suptitle('mean and error values with '+plot_legend+', observations.')
plt.tight_layout()
plt.show()

#plt.savefig('figures/plots_small2/sigma_rho_beta83-obs.png')
plt.close()

# Computing and plotting averaged errors
averaged_mean_errors = np.mean(err_mean_obs, axis=1)
averaged_mean_errors = averaged_mean_errors.reshape(10,10)

fig, ax = plt.subplots()
im = ax.pcolormesh(x,y,averaged_mean_errors, shading='auto', vmax=2., vmin=-6.)
ax.axvline(truth_x, color='red', ls='--')
ax.axhline(truth_y, color='red', ls='--')
ax.set_xlabel('sigma')
ax.set_ylabel('beta')
fig.colorbar(im, ax=ax)
plt.title('averaged error over mean values')
plt.show()
plt.close()


# Plotting STD 
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10,7))

# Plotting mean values & errors
for i in range(3) :
    im=ax[0,i].pcolormesh(x,y,stds[:,:,i], shading='auto')#, vmin=-6., vmax=4.)
    ax[0,i].axvline(truth_x, color='red', ls='--')
    ax[0,i].axhline(truth_y, color='red', ls='--')
    ax[0,i].set_xlabel('sigma')
    ax[0,i].set_ylabel('beta')
    ax[0,i].set_title('$x_{%i}$'%(i+1))
    fig.colorbar(im, ax=ax[0,i])

    im=ax[1,i].pcolormesh(x,y,errors_std[:,:,i], shading='auto', vmin=-6., vmax=4.)
    ax[1,i].axvline(truth_x, color='red', ls='--')
    ax[1,i].axhline(truth_y, color='red', ls='--')
    ax[1,i].set_xlabel('sigma')
    ax[1,i].set_ylabel('beta')
    fig.colorbar(im, ax=ax[1,i])

plt.suptitle('std and error values with '+plot_legend+', observations.')
plt.tight_layout()
plt.show()

#plt.savefig('figures/plots_small2/sigma_rho_beta83-obs.png')
plt.close()

# Computing and plotting averaged errors
averaged_std_errors = np.mean(err_std_obs, axis=1)
averaged_std_errors = averaged_std_errors.reshape(10,10)

fig, ax = plt.subplots()
im = ax.pcolormesh(x,y,averaged_std_errors, shading='auto', vmax=2., vmin=-6.)
ax.axvline(truth_x, color='red', ls='--')
ax.axhline(truth_y, color='red', ls='--')
ax.set_xlabel('sigma')
ax.set_ylabel('beta')
fig.colorbar(im, ax=ax)
plt.title('averaged error over std values')
plt.show()
plt.close()








