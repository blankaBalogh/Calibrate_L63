import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

# Loading data
fdir='../dataset/'
xt_fixed_betas = np.load(fdir+'saved_xt/170521/x_data-f-rho-beta-sigma10.npz')['arr_0']

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

x,y = np.linspace(26.5,29.5,10), np.linspace(2.,3.,10) 
#x,y = np.linspace(9.,11.,10), np.linspace(2.,3.,10)
#x,y = np.linspace(9.,11.,10), np.linspace(26.5,29.5,10)
sigmas_rhos = np.array(np.meshgrid(x, y)).T.reshape(-1,2)

comp_ind = 0
Errors = np.zeros((100,3))
Errors[:,:2] = sigmas_rhos
Errors[:,2] = error #std_pred[:,comp_ind]

log_errors = False 

vmin, vmax = 0., 1.75

if log_errors :
    Errors[:,2] = np.log(error)
    vmin, vmax = -6.,0.

Errors = np.swapaxes(Errors,0,1).reshape(3,10,10).T
errors = Errors[:,:,2]


# Plotting figure
fig = plt.figure(figsize=(6,5))

plt.pcolormesh(x, y, errors, vmin=vmin, vmax=vmax)
plt.axvline(28., color='red', ls='--')
plt.axhline(8/3, color='red', ls='--')
plt.xlabel('rho')
plt.ylabel('beta')
plt.title('error str $x_{%i}$ with sigma=10.'%(comp_ind+1))


plt.colorbar(extend='both')

#plt.show()
#plt.savefig('plots/rho_beta-STD-x%i.png'%(comp_ind+1))
fname = 'truth-rho_beta_sigma10'
if log_errors :
    fname = fname+'-log'

plt.savefig('plots/'+fname+'.png')


