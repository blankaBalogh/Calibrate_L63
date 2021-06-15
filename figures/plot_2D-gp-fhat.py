#import matplotlib
#matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import GPy


# Loading data
fdir = '../dataset/'
data_type = 'betas'
xt_fixed_betas = np.load(fdir+'030621-amix/fixed_'+data_type+'-fhat-pred-030621-larger.npz')['arr_0']
xt_fixed_betas = xt_fixed_betas.reshape(20000,100,50,6)
xt_truth = np.load(fdir+'xt_truth.npz')['arr_0']

# Computing errors
#mean_pred = np.mean(xt_fixed_betas, axis=0)[:,:3]
#mean_truth = np.mean(xt_truth, axis=0)
#err_mean = np.mean((mean_pred-mean_truth)**2, axis=1)

std_pred = np.std(xt_fixed_betas, axis=(0))[...,:3]
std_pred = np.mean(std_pred, axis=(1))
"""
indexes = np.arange(100).reshape(100,1)
thetas = xt_fixed_betas[0,:,3:]
indexes = np.arange(100).reshape(100,1)
std_pred = np.concatenate((std_pred, thetas, indexes), axis=-1)
"""

std_truth = np.std(xt_truth, axis=0)
err_std = np.mean((std_pred-std_truth)**2, axis=1)

#alpha = 1.
#error = (1.-alpha)*err_mean + alpha*err_std
error = err_std

# Fitting kriging model
train_thetas = xt_fixed_betas[0,:,0,3:]
train_errors = error.reshape(-1,1)

kernel = GPy.kern.Matern52(input_dim=3, ARD=True)

m = GPy.models.GPRegression(train_thetas, train_errors, kernel)
m.optimize(messages=True)
print(m)

# Predicting errors 
n_points = 10
x,y = np.linspace(7.,13.,n_points), np.linspace(26.,32.,n_points) 
#x,y = np.linspace(9.,11.,n_points), np.linspace(2.,3.,n_points)
#x,y = np.linspace(26.5,29.5,n_points), np.linspace(2.,3.,n_points)

sigmas_rhos = np.array(np.meshgrid(x, y)).T.reshape(-1,2)
z = np.repeat(10., n_points**2)
Thetas_to_predict = np.zeros((n_points**2,3))
Thetas_to_predict[:,0] = sigmas_rhos[:,0] #sigmas_rhos[:,0]
Thetas_to_predict[:,1] = sigmas_rhos[:,1]
Thetas_to_predict[:,2] = z #sigmas_rhos[:,1]

predicted_errors = m.predict(Thetas_to_predict)[0][:,0]

log_errors = True
if log_errors :
    min_errors = np.min(predicted_errors) - 0.001
    predicted_errors = np.log(predicted_errors - min_errors)

# Reshaping errors
#x, y =  np.linspace(9.,11.,n_points), np.linspace(2.,3.,n_points) #np.linspace(26.5,29.5,10)
#sigmas_rhos = np.array(np.meshgrid(x, y)).T.reshape(-1,2)

Errors = np.zeros((n_points**2,3))
Errors[:,:2] = sigmas_rhos
Errors[:,2] = predicted_errors

Errors = np.swapaxes(Errors,0,1).reshape(3,n_points,n_points).T
errors = Errors[:,:,2]


# Plotting figure
fig = plt.figure(figsize=(6,5))

vmin, vmax = 0., 1.75
if log_errors :
    vmin, vmax = -6.,0.

plt.pcolormesh(x,y,errors, shading='auto')#, vmin=vmin, vmax=vmax)
plt.axvline(10., color='red', ls='--')
plt.axhline(28., color='red', ls='--')
plt.xlabel('sigma')
plt.ylabel('rho')
plt.title('errors with beta=8/3.')

plt.colorbar(extend='both')

plt.show()
fname = 'gpNN-sigma-rho-beta83'
if log_errors :
    fname = fname+'-log'
#plt.savefig('plots/'+fname+'.png')

"""
import os
sdir = '/cnrm/amacs/USERS/baloghb/calibration_L63/graphiques/'+fdir
try : os.mkdir(sdir)
except : pass

plt.savefig(sdir+'sigma_beta-'+fdir[:-1]+'.png')

print(' > Successfully saved to : '+sdir+'.')
"""

