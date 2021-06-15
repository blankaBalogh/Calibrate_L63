import numpy as np
import matplotlib.pyplot as plt
import GPy
from pyDOE import lhs

from metrics import compute_loss_kriging, compute_STDloss_kriging
from scipy.optimize import minimize


fdir = 'dataset/'
xt_truth = np.load(fdir+'xt_truth.npz')['arr_0']
std_truth = np.std(xt_truth, axis=0)
et=''

# Kriging learning sample
train_std = np.load(fdir+'stds/train_stds_gp-a1.npz')['arr_0']
train_thetas = np.load(fdir+'stds/train_thetas_gp-a1.npz')['arr_0']

mean_thetas, std_thetas = np.mean(train_thetas, axis=0), np.std(train_thetas, axis=0)
mean_stds, std_stds = np.mean(train_std, axis=0), np.std(train_std, axis=0)

norm_gp = False
if norm_gp :
    et = et+'-norm'
    train_thetas = (train_thetas - mean_thetas)/std_thetas
    train_std = (train_std - mean_stds)/std_stds



# Fitting kriging model
kernel = GPy.kern.Matern52(input_dim=3, ARD=True)

m = GPy.models.GPRegression(train_thetas, train_std, kernel)
m.optimize(messages=True)
print(m)
print(m.kern.lengthscale)

# Computing m metric
def eval_metric(thetas, m, std_truth) :
    '''
    '''
    m = np.mean((m.predict(thetas)[0]-std_truth)**2)
    return m


# Optimizing theta value
n_iter = 0

def callbackF(x_) :
    '''
    Function to print all intermediate values
    '''
    global n_iter
    if n_iter%10 == 0 :
        print("Iteration no.", n_iter)
        print("theta value : ", x_)
    n_iter += 1


norms_gp = [mean_thetas, mean_stds, std_thetas, std_stds]
loss_kriging = compute_STDloss_kriging(m, std_truth, norm_gp=norm_gp, norms=norms_gp)


# list of theta IC
n_IC = 20
min_bounds, max_bounds = np.array([7.2,26.7,2.2]), np.array([12.8,29.3,2.8])
thetas_IC_list = lhs(3, samples=n_IC)*(max_bounds-min_bounds)+min_bounds
theta_stars = np.zeros((n_IC,4))*np.nan

eps, tol = 1e-3, 1e-2


for i,theta_to_update in enumerate(thetas_IC_list) :
    res = minimize(loss_kriging, theta_to_update, method='BFGS', tol=tol, 
            callback=callbackF, options={'eps':eps})
    
    theta_stars[i,:3] = res.x
    err = eval_metric(theta_stars[i,:3].reshape(1,3), m, std_truth)#[0][0,0]
    print('err ', err)
    theta_stars[i,-1] = err
print(' > optimal theta value : ', theta_stars)

#np.savez_compressed('theta_star-f-STD'+et+'.npz', theta_stars)



