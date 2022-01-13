import GPy
import numpy as np
import matplotlib.pyplot as plt
from L63_mix import Lorenz63
from data import generate_data
from pyDOE import lhs
from datetime import datetime

from scipy.optimize import minimize
from metrics import compute_loss_kriging


# If  any, orbits collapsing into a fixed point are removed.
def remove_fp(xt, last_ind=5000) :
    '''
    Replaces fixed point orbits by NaNs. 
    '''
    xt_noNans = np.copy(xt)
    std_x = np.std(xt[-last_ind:], axis=0)[...,:3]
    std_x = np.sum(std_x, axis=-1)
    fp_indexes = np.where(std_x < 6.)
    fp_theta, fp_orb = fp_indexes[0], fp_indexes[1]
    fill_nans = np.zeros((xt.shape[0], xt.shape[-1]))
    if len(fp_theta>0) :
        for i in range(len(fp_theta)) :
            xt_noNans[:,fp_theta[i],fp_orb[i]] = fill_nans
    return xt_noNans


# Defining directories, etc.
today   = datetime.today().strftime('%d%m%Y')
fdir    = 'dataset/'        # root dir
sdir    = '/Users/baloghb/Desktop/'
param   = 'sigmas'          # fixed parameter : sigma=10.
alpha   = 0.5               # see below : err_obs
datadir = fdir+today+'/'    # output directory
tag     = '-a1'             # model to load
et      = '-1d'             # model to load
log_errors = False           # log errors for final plot

# Genrating and saving f-data ? 
gen_data    = True     
save_data   = False


# Loading truth dataset
xt_truth    = np.load(fdir+'xt_truth.npz')['arr_0']
std_truth   = np.std(xt_truth, axis=0)
mean_truth  = np.mean(xt_truth, axis=0)


# -------------------------------------------------------------------------------- #
# ---------------------------------    Kriging   --------------------------------- #
# -------------------------------------------------------------------------------- #
print(' > Kriging.')
# Loading training dataset for kriging metamodel
train_thetas        = np.load(datadir+'train_thetas_gp-fhat-a1'+et+'.npz')['arr_0']
train_errors_mean   = np.load(datadir+'train_errors_mean_gp-fhat-a1'+et+'.npz')['arr_0']
train_errors_std    = np.load(datadir+'train_errors_std_gp-fhat-a1'+et+'.npz')['arr_0']

# Target variable for kriging
err_obs = (1-alpha)*train_errors_mean + alpha*train_errors_std
err_obs = err_obs.reshape(-1,1)

# Normalizing kriging input/output
mean_thetas, std_thetas = np.mean(train_thetas, axis=0), np.std(train_thetas, axis=0)
mean_y, std_y = np.mean(err_obs, axis=0), np.std(err_obs, axis=0)

norm_gp = True
if norm_gp :
    train_thetas = (train_thetas-mean_thetas)/std_thetas
    err_obs = (err_obs-mean_y)/std_y

# Fitting kriging metamodel
kernel = GPy.kern.Matern52(input_dim=1, ARD=True)
m = GPy.models.GPRegression(train_thetas, err_obs, kernel)
m.optimize(messages=True)
print(m)
print(m.kern.lengthscale)


# Predicting Errors -- final plot
mesh_len    = 20    # number of beta values
mesh_len_GP = 100   # number of beta values for the GP regressor

# Validation : sampling input betas for the kriging metamodel
Thetas_to_predict   = np.linspace(1.5,3.,mesh_len).reshape(-1,1)

if norm_gp :
    Thetas_to_predict_n = (Thetas_to_predict-mean_thetas)/std_thetas

pred_errors         = m.predict(Thetas_to_predict_n)[0][:,0]
if norm_gp :
    pred_errors = std_y*pred_errors + mean_y
    err_obs = std_y*err_obs + mean_y
    train_thetas = std_thetas*train_thetas + mean_thetas


# -------------------------------------------------------------------------------- #
# ---------------------------------    L63 model   ------------------------------- #
# -------------------------------------------------------------------------------- #
print(' > L63.')
print('   Computing errors with truth function f')
n_steps_val = 20000
spinup      = 300
x0          = np.array([0., 5., 20.])
ic          = np.zeros((mesh_len_rho*mesh_len_beta, 6))
ic[:,:3]    = np.array([x0 for i in range(mesh_len_rho*mesh_len_beta)]).reshape(-1,3)
ic[:,3]     = np.repeat(10., mesh_len_rho*mesh_len_beta)
ic[:,4:]    = Thetas_to_predict
L63         = Lorenz63()
output      = generate_data(L63, ic, n_steps=n_steps_val, dt=0.05, compute_y=False)
xt_L63      = output['x']
xt_L63      = xt_L63[spinup:]


std_L63 = np.std(xt_L63[...,:3], axis=0)
mean_L63 = np.mean(xt_L63[...,:3], axis=0)

if len(std_L63.shape) == 3 :
    std_L63 = np.mean(std_L63, axis=1)
    mean_L63 = np.mean(mean_L63, axis=1)

err_std_L63 = np.mean((std_L63-std_truth)**2, axis=1)
err_mean_L63= np.mean((mean_L63[:,:3]-mean_truth)**2, axis=1)

err_L63 = alpha*err_std_L63 + (1-alpha)*err_mean_L63



# -------------------------------------------------------------------------------- #
# ----------------------------------    Layout   --------------------------------- #
# -------------------------------------------------------------------------------- #
print(' > Plotting layout & optimization.')
if norm_gp :
    Thetas_to_predict = std_thetas*Thetas_to_predict + mean_thetas
    train_thetas = std_thetas*train_thetas + mean_thetas
    err_obs = std_err*err_obs + mean_err

rhos    = Thetas_to_predict[:,0].reshape(mesh_len_beta,mesh_len_rho)
betas   = Thetas_to_predict[:,1].reshape(mesh_len_beta,mesh_len_rho)

# If set to True, logarithms of errors will be plotted. 
# It allows a sharper representation of the function minimum. 
if log_errors :
    err_L63 = np.log(err_L63)
    err_obs = np.log(err_obs)
    min_pred= np.min(pred_errors)
    
    # min_pred can be slighly negative.
    # In this case, pred_errors are rescaled to allow the computation of the log.
    if min_pred<0 :
        eps = 1e-6
        min_pred = -min_pred+eps
        pred_errors = np.log(pred_errors+min_pred)

xlabel  = 'beta'
truth_x = 8/3


# Optimization
print('   --> Optimization.')
n_iter = 0

# Callback function for optimization (if needed)
def callbackF(x_) :
    '''
    Function to print all intermediate values
    '''
    global n_iter
    print("Iteration no.", n_iter)
    print("theta value : ", x_)
    n_iter += 1

norms_gp = np.array([mean_thetas, mean_y, std_thetas, std_y])
loss_kriging    = compute_loss_kriging(m, norm_gp=norm_gp, norms=norms_gp)
theta_to_update = [2.4]         # optimization ic
bounds = (1.5,3.2)              # bounds (if 'L-BFGS-B' optimizer is used)

# Error minimization on the kriging metamodel
print('   initial theta value : ', theta_to_update)
eps, tol = 1e-1
res = minimize(loss_kriging, theta_to_update, method='BFGS', callback=callbackF, 
        options={'eps':eps})
theta_star = res.x
print('   optimal theta value : ', theta_star,'.\n')


# Plotting layout
print('   --> Plotting layout.')

fig = plt.figure(figsize=(8,6))

plt.scatter(train_thetas[:,0], err_obs, color='skyblue', marker='*', s=5, 
        label='train kriging')
plt.scatter(x, err_L63, color='red', s=25, label='L63')
plt.plot(Thetas_to_predict, pred_errors, color='mediumblue', lw=1, label='kriging')
plt.axvline(8/3, color='red', ls='--', lw=1, label=r'$\beta_0$')
plt.axvline(theta_star, color='mediumblue', ls='--', lw=1, label=r'$\beta^*$')
plt.axvline
plt.xlabel(r'$\beta$')
plt.ylabel(r'$m$')

plt.legend()
plt.show()

