import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
from scipy.optimize import minimize
from pyDOE import lhs
from datetime import datetime

#from eL63 import embeddedLorenz63
from L63_mix import Lorenz63
from data import generate_data, generate_LHS, generate_data_solvers, generate_x0 
from metrics import *
import gpflow
from gpflow.utilities import print_summary

tag='-a1'
extra_tag = ''
exp='2d'
bias = False

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
# ---------------------------------    Kriging   --------------------------------- #
# -------------------------------------------------------------------------------- #

# Learning sample for kriging
# number of steps in learning orbits, number of orbits per theta value
dt      = 0.05      # integration time step
n_iter  = 0         # iteration no. (for callbacks)

# kriging callback : prints optimal theta value every 10 iterations
def callbackF(x_) :
    '''
    Function to print all intermediate values
    '''
    global n_iter
    if n_iter%10 == 0 :
        print("Iteration no.", n_iter)
        print("theta value : ", x_)
    n_iter += 1


n_snapshots, n_thetas = 10, 500     # number of x0 initial conditions, number of theta ic.
n_steps, spinup = 20000, 300        # orbit length, in number of integration steps

today = datetime.today().strftime('%d%m%Y') # name of saving folder
sdir = 'dataset/'+today+'/'                 # kriging learning sample will be stored here
try : os.mkdir(sdir)
except : pass

# creating or loading learning sample for kriging
print(' > Computing new learning sample for kriging.')

# sampling initial conditions 
min_bounds_x    = np.array([-25.,-25.,0.])
max_bounds_x    = np.array([25.,25.,50.])
delta_x = max_bounds_x - min_bounds_x
x0 = lhs(3, samples=n_snapshots)*delta_x + min_bounds_x
X0 = np.array([x0 for i in range(n_thetas)]).reshape(-1,3)

# sampling theta values in the learning sample
if exp=='1d' :
    min_bounds_Th   = np.array([1.5])
    max_bounds_Th   = np.array([3.2])
else :
    min_bounds_Th   = np.array([26.,1.5])
    max_bounds_Th   = np.array([32.,3.2])
    
len_thetas  = min_bounds_Th.shape[0]
delta_Th    = max_bounds_Th - min_bounds_Th
thetas_list = lhs(len_thetas, samples=n_thetas)*delta_Th + min_bounds_Th
Thetas      = np.array([[theta for i in range(n_snapshots)] for theta in thetas_list])
Thetas      = Thetas.reshape(-1, len_thetas)
betas   = np.repeat(8/3, n_thetas*n_snapshots).reshape(-1,1)
Thetas = np.concatenate((Thetas, betas), axis=-1)

ic = np.zeros((n_thetas*n_snapshots, 6))
ic[:,:3], ic[:,3:] = X0, Thetas

# Computing orbits
if bias :
    np.random.seed(42)
    bsigma  = np.random.random(1)*(11.-9.)+9.
    brho    = np.random.random(1)*(29.5-27.)+27.
    print('Biased LR parameters : sigma=%.3f, rho=%.3f.'%(bsigma, brho))
    nn_L63.sigma, nn_L63.rho = bsigma, brho

print(' > Computing output')
L63 = Lorenz63()
output = generate_data(L63, ic, n_steps=n_steps, dt=0.05, compute_y=False)
xt_pred = output['x']
xt_pred = np.array([[x[i*n_snapshots:(i+1)*n_snapshots] for i in range(n_thetas)] \
    for x in xt_pred])

# Computing errors
mean_pred = np.mean(xt_pred, axis=(0))[...,:3]
std_pred = np.std(xt_pred, axis=0)[...,:3]

if len(mean_pred.shape)==3 :
    mean_pred = np.mean(mean_pred, axis=1)
    std_pred = np.mean(std_pred, axis=1)

err_mean = np.mean((mean_pred-mean_truth)**2, axis=1)
err_std = np.mean((std_pred-std_truth)**2, axis=1)

saving_gp = True  
if bias:
    extra_tag += '-biased'

if saving_gp :
    print(' > Saving GP learning sample.')
    np.savez_compressed(sdir+'train_orbits_gp-fhat'+tag+extra_tag+'.npz',
            xt_pred)
    np.savez_compressed(sdir+'train_thetas_gp-fhat'+tag+extra_tag+'.npz', 
            thetas_list)
    np.savez_compressed(sdir+'train_errors_std_gp-fhat'+tag+extra_tag+'.npz',
            err_std)
    np.savez_compressed(sdir+'train_errors_mean_gp-fhat'+tag+extra_tag+'.npz', 
            err_mean)

else :
    print(' > Loading learning sample for kriging.')
    err_std = np.load(sdir+'train_errors_std_gp-fhat'+tag+extra_tag+'.npz')['arr_0']
    err_mean = np.load(sdir+'train_errors_mean_gp-fhat'+tag+extra_tag+'.npz')['arr_0']
    thetas_list = np.load(sdir+'train_thetas_gp-fhat'+tag+extra_tag+'.npz')['arr_0']
    len_thetas = thetas_list.shape[-1]


# Fitting of the kriging metamodel
alpha = 0.5
y = alpha*err_std + (1-alpha)*err_mean

# normalizing kriging input/target data
mean_thetas, std_thetas = np.mean(thetas_list, axis=0), np.std(thetas_list, axis=0)
mean_y, std_y = np.mean(y, axis=0), np.std(y, axis=0)

norm_gp = True
if norm_gp :
    thetas_list = (thetas_list-mean_thetas)/std_thetas
    y = (y-mean_y)/std_y

y = err_std.reshape(-1,1)

# kriging

k = gpflow.kernels.Matern52(variance=1., lengthscales=np.ones(len_thetas))
m = gpflow.models.GPR(data=(thetas_list, y), kernel=k, mean_function=None)
print_summary(m)
opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(m.training_loss, m.trainable_variables)
print_summary(m)

'''
thetas_list = thetas_list.reshape(-1, len_thetas)
kernel = GPy.kern.Matern52(input_dim=len_thetas, ARD=True)
gp = GPy.models.SparseGPRegression(thetas_list, y, kernel)
gp.optimize(messages=True)
print(gp)
print(gp.kern.lengthscale)
'''

# -------------------------------------------------------------------------------- #
# -------------------------------    Optimization   ------------------------------ #
# -------------------------------------------------------------------------------- #

print('\n -------  Optimization  -------')

norms_gp = np.array([mean_thetas, mean_y, std_thetas, std_y])
loss_kriging = compute_loss_kriging(m, norm_gp=norm_gp, norms=norms_gp)

if exp=='1d' :
    theta_to_update = [2.5]
else :
    theta_to_update = [28.5, 2.5]   # optimization starting point (or ic)

print(' > initial theta value : ', theta_to_update)

eps, tol = 1e-1, 1e-2
res = minimize(loss_kriging, theta_to_update, method='BFGS', 
        callback=callbackF, options={'eps':eps})

# optimal value of theta
theta_star = res.x

print(' > optimal theta value : ', theta_star)


### END OF SCRIPT ###
print(' > Done.')
exit()
