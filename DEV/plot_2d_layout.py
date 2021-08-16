#import matplotlib
#matplotlib.use('Agg')

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

"""
# Defining directories, etc.
today   = datetime.today().strftime('%d%m%Y')
today = '03082021'
fdir    = 'dataset/'        # root dir
param   = 'sigmas'          # fixed parameter : sigma=10.
alpha   = 0.5               # see below : err_obs
datadir = fdir+today+'/'    # output directory
tag     = '-a1'             # model to load
et      = ''                # model to load
log_errors = True           # log errors for final plot

# Genrating and saving f-data ? 
gen_data    = True     
save_data   = False
"""

# Loading truth dataset
fdir    = 'dataset/'        # root dir

xt_truth = np.load(fdir+'xt_truth.npz')['arr_0']
std_truth = np.std(xt_truth, axis=0)
mean_truth = np.mean(xt_truth, axis=0)


# Generating L63 learning sample for kriging
n_samples_k = 350
min_bounds = np.array([-35.,-35.,0.,10.,26.5,1.5])
max_bounds = np.array([35.,35.,50.,10.,32.,3.2])
delta = max_bounds - min_bounds
ic = lhs(6, samples=n_samples_k)*delta + min_bounds

n_steps_k, spinup_k = 60000, 300
L63 = Lorenz63()
output = generate_data(L63, ic, n_steps=(n_steps_k+spinup_k), dt=0.05, compute_y=False)
xt_k = output['x']
xt_k = xt_k[spinup_k:]
thetas_k = xt_k[0,:,-2:]

mean_k = np.mean(xt_k[...,:3], axis=(0))
std_k = np.std(xt_k[...,:3], axis=0)

err_std_k   = np.mean((std_k-std_truth)**2, axis=1)
err_mean_k  = np.mean((mean_k-mean_truth)**2, axis=1)

"""
norm_err_k = False
if norm_err_k :
    mean_errStd_k, std_errStd_k = np.mean(err_std_k), np.std(err_std_k)
    mean_errMean_k, std_errMean_k = np.mean(err_mean_k), np.std(err_std_k)
    mean_thetas_k, std_thetas_k = np.mean(thetas_k), np.std(thetas_k)
    err_std_kn = (err_std_k-mean_errStd_k)/std_errStd_k
    err_mean_kn = (err_mean_k-mean_errMean_k)/std_errMean_k
    thetas_kn = (thetas_k-mean_thetas_k)/std_thetas_k
else :
    err_std_kn = np.copy(err_std_k)
    err_mean_kn = np.copy(err_mean_k)
    thetas_kn = np.copy(thetas_k)
"""

alpha = 0.5
err_k = alpha*err_std_k + (1.-alpha)*err_mean_k


# kriging
err_k = err_k.reshape(-1,1)
kernel = GPy.kern.Matern32(input_dim=2, ARD=True)
m = GPy.models.GPRegression(thetas_k, err_k, kernel)
m.optimize(messages=True)
print(m)
print(m.kern.lengthscale)

mesh_len = 20
x, y = np.linspace(26.,32.,mesh_len), np.linspace(1.5,3.2,mesh_len)
sigmas_rhos = np.array(np.meshgrid(x, y)).T.reshape(-1,2)
Thetas_to_predict = np.copy(sigmas_rhos)

# Predicting errors for the final mesh plot
pred_errors = m.predict(Thetas_to_predict)[0][:,0]


# L63 model on the same mesh
# sampling x initial conditions
n_snapshots = 10
min_bounds, delta = min_bounds[:3], delta[:3]
x0 = lhs(3, samples=n_snapshots)*delta + min_bounds
x0 = np.array(list(x0)*mesh_len**2)
ic_thetas = np.repeat(Thetas_to_predict.T,n_snapshots).reshape(2,-1).T
ic_sigmas = np.repeat(10., n_snapshots*mesh_len**2)

ic = np.zeros((n_snapshots*mesh_len**2,6))*np.nan
ic[...,:3]  = x0
ic[...,3]   = ic_sigmas
ic[...,4:]  = ic_thetas

output_L63 = generate_data(L63, ic, n_steps=(n_steps_k+spinup_k), dt=0.05, compute_y=False)
xt_L63 = output_L63['x']
xt_L63 = xt_L63[spinup_k:]
# long !
xt_L63 = np.array([xt_L63[:,i*n_snapshots:(i+1)*n_snapshots] for i in range(mesh_len**2)])
xt_L63 = np.swapaxes(xt_L63,0,1)

std_L63 = np.mean(np.std(xt_L63[...,:3], axis=0), axis=1)
mean_L63 = np.mean(xt_L63[...,:3], axis=(0,2))

err_std_L63 = np.mean((std_L63-std_truth)**2, axis=1)
err_mean_L63 = np.mean((mean_L63-mean_truth)**2, axis=1)
err_L63 = alpha*err_std_L63 + (1.-alpha)*err_mean_L63


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

log_errors = True
if log_errors :
    err_L63 = np.log(err_L63)
    min_pred= np.min(pred_errors)
    if min_pred<0 :
        eps = 1e-6
        min_pred = -min_pred+eps
    pred_errors = np.log(pred_errors+min_pred)

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

norms_gp = np.array([mean_thetas, mean_y, std_thetas, std_y], dtype=object)

loss_kriging    = compute_loss_kriging(m, norm_gp=norm_gp, norms=norms_gp)
theta_to_update = [27., 2.4]        # optimization ic
bounds = ((26.5,32.),(1.5,3.2))     # bounds (if 'L-BFGS-B' optimizer is used)

print('   initial theta value : ', theta_to_update)
eps, tol = 1e-1, 1e-3
res = minimize(loss_kriging, theta_to_update, method='BFGS', callback=callbackF, 
        options={'eps':eps})
theta_star = res.x
print('   optimal theta value : ', theta_star,'.\n')


# Plotting layouts
today = '12082021'
et = ''
print('   --> Plotting layout.')
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,7))

im=ax[0].pcolormesh(x,y,errors_L63[:,:,0], shading='auto')#, vmin=-3., vmax=1.5)
ax[0].axvline(truth_x, color='red', ls='--')
ax[0].axhline(truth_y, color='red', ls='--')
ax[0].set_xlabel(xlabel)
ax[0].set_ylabel(ylabel)
ax[0].set_title('m (obtained with $f$)')
fig.colorbar(im, ax=ax[0])


im=ax[1].pcolormesh(x,y,errors_NN[:,:,0], shading='auto', vmin=-3., vmax=1.5)
ax[1].axvline(truth_x, color='red', ls='--')
ax[1].axhline(truth_y, color='red', ls='--')
ax[1].set_xlabel(xlabel)
ax[1].set_ylabel(ylabel)
ax[1].set_title('$\hat{m}$ (obtained with $\hat{f}$)')
fig.colorbar(im, ax=ax[1])

im=ax[2].pcolormesh(x,y,errors_pred[:,:,0], shading='auto')#, vmin=-3., vmax=1.5)
ax[2].axvline(truth_x, color='red', ls='--')
ax[2].axhline(truth_y, color='red', ls='--')
ax[2].set_xlabel(xlabel)
ax[2].set_ylabel(ylabel)
ax[2].set_title('kriged m (kriging on $\hat{f}$)')
fig.colorbar(im, ax=ax[2])

plt.tight_layout()
plt.show()
#plt.savefig('figures/layout-fhat-fhatkrig-'+today+et+'.png')

