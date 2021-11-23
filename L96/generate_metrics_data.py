import os
import numpy as np
import time
import GPy
from scipy.optimize import minimize
from pyDOE import lhs
from datetime import datetime

from L96 import Lorenz96
from ML_model import ML_model, r2_score_keras
from data import generate_data 
from metrics import *

# Sharing available GPU resources
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus :
    tf.config.experimental.set_memory_growth(gpu, True)

tag         = '-a2'     # NN learing sample
extra_tag   = ''        # specific name of the NN learning sample
exp         = '1d'      # '2d' or '1d'

if exp=='2d' :
    extra_tag += '-2d'
    ndim = 1
else :
    ndim = 2


# -------------------------------------------------------------------------------- #
# ----------------    2nd EXPERIMENT : LEARNING dx = f(x,theta)   ---------------- #
# -------------------------------------------------------------------------------- #
print(' > Loading learning samples. ')

# Setting up NN model
layers = [32,32]
n_epochs    = 30
dic_NN      = {'name':'f_lhs', 'in_dim':2, 'out_dim':1, 'nlays':layers, 
    'dropout':False, 'exp':exp}

# Loading model 1
nn_L96      = ML_model(dic_NN)
print(nn_L96.model.summary())


learning_rate = 1e-3
K,J = 8,32
loss_fn = tf.keras.losses.MeanSquaredError()        # loss function. Can be custom. 
optim   = tf.keras.optimizers.Adam(learning_rate)

nn_L96.model.compile(loss=loss_fn, optimizer=optim, metrics=[r2_score_keras])

print(' > Loading model weights.')
nn_L96.model.load_weights('weights/best-weights-b0.h5')

"""
# test
n_steps, n_ic = 20, 5
h0,c0,F0,b0 = 1.,10.,20.,9.5
dict_L96 = {'h':h0,'c':c0,'b':b0,'F':F0,'K':K,'J':J,'nn_model':nn_L96}
L96 = Lorenz96(dict_L96)
np.random.seed(42)
x0 = np.random.random((n_ic,K+K*J+4))
x0[...,-4:] = np.repeat([h0,c0,F0,b0],n_ic).reshape(4,n_ic).T

L96.mode = 'full_nn'
output = generate_data(L96,x0=x0,n_steps=n_steps,dt=0.01,compute_y=False)

L96.mode='lorenz'
output_lorenz = generate_data(L96,x0,n_steps=n_steps,dt=0.01,compute_y=False)

# end test
"""
# -------------------------------------------------------------------------------- #
# -----------------------------    Generating data   ----------------------------- #
# -------------------------------------------------------------------------------- #
h0,c0,F0,b0 = 1.,10.,20.,10.
dict_L96 = {'h':h0,'c':c0,'b':b0,'F':F0,'K':K,'J':J,'nn_model':nn_L96}
L96 = Lorenz96(dict_L96)

# Generating 'truth' dataset
L96.mode = 'lorenz'
mesh_len_gp = 1500
#Thetas_to_predict = np.linspace(5.,15.,mesh_len_gp)

min_bounds = np.array([-1]*(K*(J+1))+[h0,b0,F0,b0])
max_bounds = np.array([1]*(K*(J+1))+[h0,b0,F0,b0])
ic = np.random.random((mesh_len_gp,K+K*J+4))*(max_bounds-min_bounds)+min_bounds
#ic = np.random.random((mesh_len_gp, K+K*J+4))
#ic[...,-4:-1] = np.repeat([h0,c0,F0],mesh_len_gp).reshape(3,mesh_len_gp).T
#ic[...,-1] = Thetas_to_predict

#output_lz = generate_data(L96,x0=ic,n_steps=n_steps,dt=0.01,compute_y=False)


#ic = np.random.random((mesh_len_gp, K+K*J+4))
#ic[...,-4:-1] = np.repeat([h0,c0,F0],mesh_len_gp).reshape(3,mesh_len_gp).T
#ic[...,-1] = Thetas_to_predict

n_steps = 500
L96.mode = 'full_nn'
output_nn = generate_data(L96,x0=ic,n_steps=n_steps,dt=0.05,compute_y=False)

#L96.mode='lorenz'
#output_lz = generate_data(L96,x0=ic,n_steps=2000,dt=0.01,compute_y=False)


# kriging
# Generating learning sample
xt_truth = np.load('data/xt_truth.npz')['arr_0']
xt_nn = output_nn['x'][:,:,:K]
mean_truth = np.mean(xt_truth[...,:K])
thetas = output_nn['x'][0,:,-1]
mean_pred = np.mean(xt_nn, axis=(0,-1))
std_pred = np.mean(np.std(xt_nn,axis=0),axis=(-1))
std_truth = np.mean(np.std(xt_truth,axis=0))

errors = np.square(std_pred-std_truth).reshape(-1,1)
err_mean = np.square(mean_pred-mean_truth).reshape(-1,1)
thetas = ic[:,-1].reshape(-1,1)

kernel = GPy.kern.Matern52(input_dim=1)
m = GPy.models.GPRegression(thetas, errors, kernel)
m.optimize(messages=True)
print(m)
print(m.kern.lengthscale)



b_to_update = 10.1

Thetas_to_predict = np.linspace(5.,15.,100)
m_tilde = m.predict(Thetas_to_predict.reshape(-1,1))[0][:,0]

"""
fig,ax=plt.subplots(ncols=2,figsize=(10,6))

ax[0].plot(Thetas_to_predict,m_tilde_mean,label='kriging',color='mediumblue')
ax[0].scatter(thetas,err_mean,marker='+',s=1,color='skyblue',label='nn')
ax[0].scatter(thetas,err_mean_lz,marker='o',s=2,color='red',label='l96')
ax[0].set_xlabel(r'$b$')
ax[0].set_ylabel(r'$m$')
ax[0].axvline(10.,color='k',ls='--',lw=1,label=r'$b_0$')
plt.legend()
ax[0].set_title('mean')

ax[1].plot(Thetas_to_predict,m_tilde,label='kriging',color='mediumblue')
ax[1].scatter(thetas,errors,marker='+',s=1,color='skyblue',label='nn')
ax[1].scatter(thetas,err_std_lz,marker='o',s=2,color='red',label='l96')
ax[1].set_xlabel(r'$b$')
ax[1].axvline(10.,color='k',ls='--',lw=1,label=r'$b_0$')
ax[1].set_ylabel(r'$m$')
plt.legend()
ax[1].set_title('std')


plt.show()
"""


# optimization

def loss_kriging(m) :
    def loss_fun(th) :
        theta_ = np.array([th])
        err = m.predict(theta_)[0][:,0]
        return err[0]
    return loss_fun

b_to_update = np.array([9.4])
loss_kr = loss_kriging(m)
res = minimize(loss_kr, b_to_update, method='BFGS', options={'eps':1e-4})
print(' > optimal b : ', res.x)


