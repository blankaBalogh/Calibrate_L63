import os
import GPy
import numpy as np
from pyDOE import lhs
from scipy.optimize import minimize
from datetime import datetime

#from eL63 import embeddedLorenz63
from L63_mix import Lorenz63
from ML_model_param import ML_model, train_ML_model
from data import generate_data, generate_LHS, generate_data_solvers, generate_x0 
from metrics import *

# Sharing available GPU resourcesimport tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus :
    tf.config.experimental.set_memory_growth(gpu, True)


# Parsing aruments
from argparse import ArgumentParser

parser = ArgumentParser()
# Type of learning sample : a=1 means LHS, a=2 means 'orbit'.
parser.add_argument('-a', '--learning_sample', default=1,
        help='Learning sample selection : orbit (=2) or lhs (=1).')
# Extra tag that will be added to the output file name.
parser.add_argument('-et', '--extra_tag', type=str, default='', 
        help='Adds an extra tag. Useful to save new datasets.')
# If parser, generates a new learning sample for the kriging metamodel
parser.add_argument('-gp', '--new_gp', default=False, action='store_true',
        help='If parsed, generates a new learning sample for the kriging model.')
# Experience type : '1d' or '2d'.
parser.add_argument('-exp', '--experience', type=str, default='2d',
        help="Experience type : '2d' or '1d'.")
# If parsed, adds random bias to sigma and rho variables, and sets exp='1d'.
parser.add_argument('-b', '--biased_lr', default=False, action='store_true', 
        help='If parsed, sigma and rho values are randomly sampled in LR experiment. \
                Available with exp=1d only.')

args = parser.parse_args()

tag         = '-a'+str(args.learning_sample)
extra_tag   = args.extra_tag
new_gp      = args.new_gp
exp         = args.experience
bias        = args.biased_lr

if exp=='1d' :
    extra_tag += '-1d'

if bias :
    exp='1d'

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
# ----------------------------    Loading NN model   ----------------------------- #
# -------------------------------------------------------------------------------- #
print(' > Loading learning samples.')

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


# Normalization of x & y data
norm_input = True

if norm_input :
    mean_x, std_x = np.mean(x_data, axis=0), np.std(x_data, axis=0)
    mean_y, std_y = np.mean(y_data, axis=0), np.std(y_data, axis=0)
    x_data = (x_data-mean_x)/std_x
    y_data = (y_data-mean_y)/std_y

# Selecting relevant predictors depending on the experience type
if exp=='2d' : 
    liste_dims = [0,1,2,4,5]

else :
    liste_dims = [0,1,2,5]
    y_data = y_data[:,-1]
    y_data = y_data.reshape(-1,1)
    mean_y, std_y = mean_y[-1], std_y[-1]

x_data = x_data[...,liste_dims]
if norm_input :
    mean_x, std_x = mean_x[liste_dims], std_x[liste_dims]

# Setting up NN model
if exp=='2d' :
    layers = [1024, 512, 256, 128, 64, 32, 16]
else :
    layers = [256, 128, 64, 32, 16]

dic_NN = {'name':'f_orb', 'in_dim':x_data.shape[1], 'out_dim':y_data.shape[1], 
        'nlays':layers, 'dropout':False, 'exp':exp}
nn_L63          = ML_model(dic_NN)
nn_L63.norms    = [mean_x, mean_y, std_x, std_y]
nn_L63.suffix   = tag+extra_tag
print(nn_L63.model.summary())

# Loading best model weights
print(' > Loading model weights.')
nn_L63.model.load_weights('weights/best-weights'+nn_L63.suffix+'.h5')




# -------------------------------------------------------------------------------- #
# ---------------------------------    Kriging   --------------------------------- #
# -------------------------------------------------------------------------------- #

# Learning sample for kriging
dt      = 0.05      # integration time step
n_iter  = 0         # iteration no. (for callbacks)

# Callbacks for loss minimzation 
def callbackF(x_) :
    '''
    Function to print all intermediate values
    '''
    global n_iter
    if n_iter%10 == 0 :
        print("Iteration no.", n_iter)
        print("theta value : ", x_)
    n_iter += 1

n_samples_kriging = 750
n_steps, spinup = 20000, 300        # orbit length, in number of integration steps

today = datetime.today().strftime('%d%m%Y') # name of saving folder
sdir = 'dataset/'+today+'/'                 # kriging learning sample will be stored here
try : os.mkdir(sdir)
except : pass

# creating or loading learning sample for kriging
if new_gp :
    print(' > Computing new learning sample for kriging.')

    # sampling theta values in the learning sample
    if exp=='1d' :
        min_bounds_Th   = np.array([-25.,-25.,0.,10.,28.,1.5])
        max_bounds_Th   = np.array([25.,25.,50.,10.,28.,3.])
        len_thetas = 1
    else :
        min_bounds_Th   = np.array([-25.,-25.,0.,10.,26.5,1.5])
        max_bounds_Th   = np.array([25.,25.,50.,10.,32.,3.])
        len_thetas = 2
    
    delta_Th    = max_bounds_Th - min_bounds_Th

    ic = lhs(6,samples=n_samples_kriging)*delta_Th + min_bounds_Th
    thetas_list = ic[:,-len_thetas:]

    # Computing orbits
    if bias :
        np.random.seed(42)
        bsigma  = np.random.random(1)*(11.-9.)+9.
        brho    = np.random.random(1)*(29.5-27.)+27.
        print('Biased LR parameters : sigma=%.3f, rho=%.3f.'%(bsigma, brho))
        nn_L63.sigma, nn_L63.rho = bsigma, brho
    
    # Generating validation orbits with the NN model
    print(' > Computing output')
    output = generate_data(nn_L63, ic, n_steps=n_steps+spinup, dt=0.05, compute_y=False)
    xt_pred = output['x']

    # Removing spinup
    xt_pred = xt_pred[spinup:]

    # Computing errors
    mean_pred = np.mean(xt_pred, axis=(0))[...,2]
    std_pred = np.std(xt_pred, axis=0)[...,:3]

    if len(std_pred.shape)==3 :
        mean_pred = np.mean(mean_pred, axis=1)
        std_pred = np.mean(std_pred, axis=1)

    # Computing prediction errors over the validation orbits
    err_mean = (mean_pred-mean_truth[2])**2
    print('mean error : ', np.mean(err_mean))
    err_std = 2*np.sum((std_pred-std_truth)**2, axis=1)
    print('std error : ', np.mean(err_std))
    
    alpha = 0.5         # selecting 'mean' or 'std' errors
    err = alpha*err_std + (1.-alpha)*err_mean
    print('Removing outliers.')    
        
    saving_gp = True  
    if bias:
        extra_tag += '-biased'

    # Saving kriging learning sample
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

# If optimization is performed over an existing learning sample
else :
    print(' > Loading learning sample for kriging.')
    err_std = np.load(sdir+'train_errors_std_gp-fhat'+tag+extra_tag+'.npz')['arr_0']
    err_mean = np.load(sdir+'train_errors_mean_gp-fhat'+tag+extra_tag+'.npz')['arr_0']
    thetas_list = np.load(sdir+'train_thetas_gp-fhat'+tag+extra_tag+'.npz')['arr_0']
    len_thetas = thetas_list.shape[-1]


# Fitting of the kriging metamodel
alpha = 0.5
y = alpha*err_std + (1-alpha)*err_mean      # kriging target variable : val. orbit errors

# normalizing kriging input/target data
mean_thetas, std_thetas = np.mean(thetas_list, axis=0), np.std(thetas_list, axis=0)
mean_y, std_y = np.mean(y, axis=0), np.std(y, axis=0)

norm_gp = False
if norm_gp :
    thetas_list = (thetas_list-mean_thetas)/std_thetas
    y = (y-mean_y)/std_y

y = err_std.reshape(-1,1)

# kriging
kernel = GPy.kern.Matern52(input_dim=len_thetas, ARD=True)
m = GPy.models.GPRegression(thetas_list, y, kernel)
m.optimize(messages=True)
print(m)
print(m.kern.lengthscale)


# -------------------------------------------------------------------------------- #
# -------------------------------    Optimization   ------------------------------ #
# -------------------------------------------------------------------------------- #

print('\n> Optimization')

norms_gp = np.array([mean_thetas, mean_y, std_thetas, std_y])
# Loading and parameterizing kriging loss function from metrics
loss_kriging = compute_loss_kriging(m, norm_gp=norm_gp, norms=norms_gp)

if exp=='1d' :
    theta_to_update = [2.5]
else :
    theta_to_update = [28.5, 2.5]   # optimization starting point (or ic)

print(' > initial theta value : ', theta_to_update)

eps = [1e-1, 1e-2]
res = minimize(loss_kriging, theta_to_update, method='BFGS', 
        callback=callbackF, options={'eps':eps})


# optimal value of theta
theta_star = res.x

print(' > optimal theta value : ', theta_star)


### END OF SCRIPT ###
print('\n> Done.')
exit()
