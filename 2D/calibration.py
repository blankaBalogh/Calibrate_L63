import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
sns.set_style('white')
from scipy.optimize import minimize
from pyDOE import lhs
from datetime import datetime

#from eL63 import embeddedLorenz63
from L63_mix import Lorenz63
from ML_model_param import ML_model, train_ML_model
from data import generate_data, generate_LHS, generate_data_solvers, generate_x0 
from metrics import *
import GPy

# Sharing GPU resources (compat. TF>2.0)
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus :
    tf.config.experimental.set_memory_growth(gpu, True)


# Parsing aruments
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-a', '--learning_sample', default=1,
        help='Learning sample selection : orbit (=2) or lhs (=1).')
parser.add_argument('-et', '--extra_tag', type=str, default='', 
        help='Adds an extra tag. Useful to save new datasets.')
parser.add_argument('-gp', '--new_gp', default=False, action='store_true')

args    = parser.parse_args()

tag = '-a'+str(args.learning_sample)
extra_tag = args.extra_tag
new_gp = args.new_gp

if tag=='-a1' : learning_sample = 'LHS'
else : learning_sample = 'orbits'

print('\n > Learning sample : %s.'%learning_sample)

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
# ----------------    2nd EXPERIMENT : LEARNING dx = f(x,theta)   ---------------- #
# -------------------------------------------------------------------------------- #
print('***** 2nd experiment : learning to predict full y_data. ***** ')
print(' ------- Loading learning samples ------- ')

# Loading 'orbits' learning sample
if tag=='-a2' :
    print(' > Loading learning sample of orbits.')
    x_data = np.load('dataset/x_data-a2'+extra_tag+'.npz')['arr_0']
    y_data = np.load('dataset/y_data-a2'+extra_tag+'.npz')['arr_0'][...,:3]
    x_data, y_data = np.swapaxes(x_data,0,1), np.swapaxes(y_data,0,1) 
    x_data = x_data.reshape(-1, x_data.shape[-1])
    y_data = y_data.reshape(-1, y_data.shape[-1])

# Loading lhs learning sample
if tag=='-a1' :
    print(' > Loading learning sample of LHS sample.')
    x_data = np.load('dataset/x_data-a1'+extra_tag+'.npz')['arr_0'][0]
    y_data = np.load('dataset/y_data-a1'+extra_tag+'.npz')['arr_0'][0][...,:3]
    x_data = np.delete(x_data, 3, axis=-1)


# Learning fhat_betas
print('\n ------ Learning fhat_thetas ------- ')

# Normalization of x & y data
mean_x, std_x = np.mean(x_data, axis=0), np.std(x_data, axis=0)
mean_y, std_y = np.mean(y_data, axis=0), np.std(y_data, axis=0)
x_data = (x_data-mean_x)/std_x
y_data = (y_data-mean_y)/std_y

layers = [1024, 512, 256, 128, 64, 32, 16]

print('y data shape : ', y_data.shape)

dic_NN = {'name':'f_orb', 'in_dim':x_data.shape[1], 'out_dim':y_data.shape[1], 'nlays':layers}
nn_L63 = ML_model(dic_NN)
nn_L63.norms = [mean_x, mean_y, std_x, std_y]
nn_L63.suffix = tag+extra_tag
nn_L63.name = 'model'+tag+extra_tag
print(' > Model to load : %s.'%nn_L63.name)
print(nn_L63.model.summary())

# Loading best weights
print(' > Loading model weights.')
nn_L63.model.load_weights('weights/best-weights'+nn_L63.suffix+'.h5')



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


n_snapshots, n_thetas = 25, 350     # number of x0 initial conditions, number of theta ic.
n_steps, spinup = 20000, 300        # orbit length, in number of integration steps

today = datetime.today().strftime('%d%m%Y') # name of saving folder
sdir = 'dataset/'+today+'/'                 # kriging learning sample will be stored here
try : os.mkdir(sdir)
except : pass

# creating or loading learning sample for kriging
if new_gp :
    print(' > Computing new learning sample for kriging.')

    # sampling initial conditions 
    min_bounds_x    = np.array([-25.,-25.,0.])
    max_bounds_x    = np.array([25.,25.,50.])
    delta_x = max_bounds_x - min_bounds_x
    x0 = lhs(3, samples=n_snapshots)*delta_x + min_bounds_x
    X0 = np.array([x0 for i in range(n_thetas)]).reshape(-1,3)

    # sampling theta values in the learning sample
    min_bounds_Th   = np.array([26.5,1.5])
    max_bounds_Th   = np.array([32.,3.2])
    delta_Th = max_bounds_Th - min_bounds_Th
    thetas_list = lhs(2, samples=n_thetas)*delta_Th + min_bounds_Th
    Thetas = np.array([[theta for i in range(n_snapshots)] for theta in thetas_list])
    Thetas = Thetas.reshape(-1,2)

    ic = np.zeros((n_thetas*n_snapshots,5))
    ic[:,:3], ic[:,3:] = X0, Thetas

    # Computing orbits 
    print(' > Computing output')
    output = generate_data(nn_L63, ic, n_steps=n_steps, dt=0.05, compute_y=False)
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
    extra_tag = extra_tag
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


# Fitting of the kriging metamodel
alpha = 0.5
y = alpha*err_std + (1-alpha)*err_mean

# normalizing kriging input/target data
mean_thetas, std_thetas = np.mean(thetas_list, axis=0), np.std(thetas_list, axis=0)
mean_y, std_y = np.mean(err_std, axis=0), np.std(err_std, axis=0)

norm_gp = True
if norm_gp :
    thetas_list = (thetas_list-mean_thetas)/std_thetas
    y = (y-mean_y)/std_y

y = err_std.reshape(-1,1)

# kriging
thetas_list = thetas_list.reshape(-1,2)
kernel = GPy.kern.Matern52(input_dim=2, ARD=True)
gp = GPy.models.GPRegression(thetas_list, y, kernel)
gp.optimize(messages=True)
print(gp)
print(gp.kern.lengthscale)


# -------------------------------------------------------------------------------- #
# -------------------------------    Optimization   ------------------------------ #
# -------------------------------------------------------------------------------- #

print('\n -------  Optimization  -------')

norms_gp = np.array([mean_thetas, mean_y, std_thetas, std_y])
loss_kriging = compute_loss_kriging(gp, norm_gp=norm_gp, norms=norms_gp)
    
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
