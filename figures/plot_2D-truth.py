import numpy as np
import matplotlib.pyplot as plt

# Loading data
fdir='../dataset/'
xt_fixed_betas = np.load(fdir+'truth/x_data-sigma-beta-rho28.npz')['arr_0']

xt_truth = np.load(fdir+'xt_truth.npz')['arr_0']


# Computing errors
mean_pred = np.mean(xt_fixed_betas, axis=0)[:,:3]
mean_truth = np.mean(xt_truth, axis=0)
err_mean = np.mean((mean_pred-mean_truth)**2, axis=1)

std_pred = np.std(xt_fixed_betas, axis=0)[:,:3]
std_truth = np.std(xt_truth, axis=0)
err_std = np.mean((std_pred-std_truth)**2, axis=1)

alpha = 1.
error = (1.-alpha)*err_mean + alpha*err_std



x, y = np.linspace(9.,11.,10), np.linspace(2.,3.,10) #,np.linspace(26.5,29.5,10)
sigmas_rhos = np.array(np.meshgrid(x, y)).T.reshape(-1,2)

Errors = np.zeros((100,3))
Errors[:,:2] = sigmas_rhos
Errors[:,2] = np.arange(100) #error

Errors = np.swapaxes(Errors,0,1).reshape(3,10,10).T
errors = Errors[:,:,2]


# Plotting figure
fig = plt.figure(figsize=(6,5))

plt.pcolormesh(x, y, errors)#, vmin=0., vmax=1.)
plt.axvline(10., color='red', ls='--')
plt.axhline(8/3, color='red', ls='--')
plt.xlabel('rho')
plt.ylabel('beta')
plt.title('errors with rho=28.')

plt.colorbar(extend='both')

plt.show()


