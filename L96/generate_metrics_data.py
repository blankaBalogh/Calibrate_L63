import os
import numpy as np
import time
import GPy
from scipy.optimize import minimize
from pyDOE import lhs
from datetime import datetime
import time

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

from argparse import ArgumentParser

parser = ArgumentParser()
# Type of learning sample : a=1 means LHS, a=2 means 'orbit'.
parser.add_argument('-F', '--F0', default=10., type=float, 
        help='Value of F parameter.')
args = parser.parse_args()
F0 = args.F0

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
nn_L96.norms = None
print(nn_L96.model.summary())


learning_rate = 1e-3
K,J = 8,32
loss_fn = tf.keras.losses.MeanSquaredError()        # loss function. Can be custom. 
optim   = tf.keras.optimizers.Adam(learning_rate)

nn_L96.model.compile(loss=loss_fn, optimizer=optim, metrics=[r2_score_keras])

print(' > Loading model weights.')
nn_L96.model.load_weights('weights/best-weights.h5')


# -------------------------------------------------------------------------------- #
# -----------------------------    Generating data   ----------------------------- #
# -------------------------------------------------------------------------------- #
h0,c0,b0 = 1.,10.,10.
K = 8
print(' > F0 : ', F0)
dict_L96 = {'h':h0,'c':c0,'b':b0,'F':F0,'K':K,'J':J,'nn_model':nn_L96}
L96 = Lorenz96(dict_L96)

# Generating L96 datasets
L96.mode = 'lorenz'
mesh_len_gp = 250
n_steps = 3000

min_bounds = np.array([-1]*(K*(J+1))+[h0,6.,F0,b0])
max_bounds = np.array([1]*(K*(J+1))+[h0,14.,F0,b0])
ic = lhs((K+K*J+4),mesh_len_gp)*(max_bounds-min_bounds)+min_bounds

# Kriging learning sample : NN validation orbits
L96.mode = 'full_nn'
print(' > Generating kriging learning sample.')
ts = time.time()
output_nn = generate_data(L96,x0=ic,n_steps=n_steps,dt=0.005,compute_y=False)
te = time.time()
elapsed = te-ts
print(' > Elapsed time : %.3f s.'%elapsed)
x = output_nn['x']
y = output_nn['y']
np.savez_compressed('data/x_nn_F'+str(F0)+'.npz',x)
np.savez_compressed('data/y_nn_F'+str(F0)+'.npz',y)


# 'Truth' orbits : LZ equations
print(' > Generating Lorenz orbits from the same IC.')
L96.mode = 'lorenz'
output_lz = generate_data(L96,x0=ic,n_steps=n_steps,dt=0.005,compute_y=False)
x_lz, y_lz = output_lz['x'], output_lz['y']
np.savez_compressed('data/x_lz_F'+str(F0)+'.npz',x_lz)
np.savez_compressed('data/y_lz_F'+str(F0)+'.npz',y_lz)


# kriging
# Loading & reshaping learning sample
xt_truth = np.load('data/xt_truth.npz')['arr_0'][...,:K]
xt_nn = np.load('data/x_nn_F'+str(F0)+'.npz')['arr_0']
thetas = np.copy(xt_nn)[0,:,-3]
xt_nn = xt_nn[...,:K]

# Computing long-term metrics & errors
mean_pred = np.mean(xt_nn, axis=(0,-1))
std_pred = np.mean(np.std(xt_nn,axis=0),axis=(-1))
mean_truth = np.mean(xt_truth)
std_truth = np.mean(np.std(xt_truth,axis=0))

errors = np.square(std_pred-std_truth).reshape(-1,1)
err_mean = np.square(mean_pred-mean_truth).reshape(-1,1)
thetas = thetas.reshape(-1,1)

# Fitting the kriging model
kernel = GPy.kern.Matern52(input_dim=1)
m = GPy.models.GPRegression(thetas, errors, kernel)
m.optimize(messages=True)
print(m)
print(m.kern.lengthscale)


# Finding the optimal value of c

def loss_kriging(m) :
    def loss_fun(th) :
        theta_ = np.array([th])
        err = m.predict(theta_)[0][:,0]
        return err[0]
    return loss_fun

b_to_update = np.array([9.4])
loss_kr = loss_kriging(m)
res = minimize(loss_kr, b_to_update, method='BFGS', options={'eps':1e-4})
print(' > optimal c : ', res.x)


# Fig 1
plt.scatter(c,err_nn,s=7,marker='+',label='learning sample')
plt.plot(thetas_to_pred, err_k[0], label='kriging',color='mediumblue')
plt.xlabel(r'$\theta=c$')
plt.ylabel(r'MSE')
plt.axvline(10.,color='k',ls='--',lw=1,label=r'$c_0=10.$')
plt.axvline(cstar,color='mediumblue',ls='--',lw=1,label=r'$c^*=%.3f$'%cstar)
plt.legend()
plt.show()

