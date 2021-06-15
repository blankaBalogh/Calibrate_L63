import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

# Loading data
fdir = '../dataset/'
#fdir = '/cnrm/amacs/USERS/baloghb/calibration_L63/v6/exp2/dataset/'

xt_fixed_betas = np.load(fdir+'270521/fixed_betas-fhat-pred-310521-larger.npz')['arr_0']
xt_fixed_betas = np.array([[x[i*50:(i+1)*50] for i in range(100)] for x in xt_fixed_betas])
# Resulting array of shape : (20000,100,50,6).

xt_truth = np.load(fdir+'xt_truth.npz')['arr_0']

# Computing errors
std_pred = np.std(xt_fixed_betas, axis=(0))[:,:,:3]
std_pred = np.mean(std_pred, axis=(1))

mean_pred = np.mean(xt_fixed_betas, axis=(0,2))

"""
indexes = np.arange(100).reshape(100,1)
thetas = xt_fixed_betas[0,:,3:]
indexes = np.arange(100).reshape(100,1)
std_pred = np.concatenate((std_pred, thetas, indexes), axis=-1)
"""

std_truth = np.std(xt_truth, axis=0)
err_std = np.mean((std_pred-std_truth)**2, axis=1)

alpha = 1.
error = err_std #(1.-alpha)*err_mean + alpha*err_std


#x,y = np.linspace(7.,13.,10), np.linspace(2.,3.,10) 
x,y = np.linspace(7.,13.,10), np.linspace(25.,32.,10)
sigmas_rhos = np.array(np.meshgrid(x, y)).T.reshape(-1,2)


Errors = np.zeros((100,3))
Errors[:,:2] = sigmas_rhos
Errors[:,2] = error

Errors = np.swapaxes(Errors,0,1).reshape(3,10,10).T
errors = Errors[:,:,2]


# Plotting figure
fig = plt.figure(figsize=(6,5))

log_errors = True
if log_errors :
    errors = np.log(errors)


plt.pcolormesh(x,y,errors, shading='auto')#, vmin=0., vmax=1.2)
plt.axvline(10., color='red', ls='--')
plt.axhline(28., color='red', ls='--')
plt.xlabel('sigma')
plt.ylabel('rho')
plt.title('errors with rho=28.')

plt.colorbar(extend='both')

#plt.show()
plt.savefig('plots_310521/sigma_rho_beta83-15e5Samples.png')
"""
import os
sdir = '/cnrm/amacs/USERS/baloghb/calibration_L63/graphiques/'+fdir
try : os.mkdir(sdir)
except : pass

plt.savefig(sdir+'sigma_beta-'+fdir[:-1]+'.png')

print(' > Successfully saved to : '+sdir+'.')
"""

