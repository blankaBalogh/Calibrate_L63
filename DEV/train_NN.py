import numpy as np
from pyDOE import lhs

from L63_mix import Lorenz63
from ML_model_param import ML_model, train_ML_model
from data import generate_data, generate_LHS, generate_data_solvers, generate_x0 
from metrics import *

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus :
    tf.config.experimental.set_memory_growth(gpu, True)

import matplotlib.pyplot as plt


# -- Parsing aruments
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-a', '--learning_sample', default=1,
        help='Learning sample selection : orbit (=2) or lhs (=1).')
parser.add_argument('-et', '--extra_tag', type=str, default='', 
        help='Adds an extra tag. Useful to save new datasets.')
parser.add_argument('-exp', '--experience', type=str, default='2d',
        help="Experience type : '2d' or '1d'.")

args    = parser.parse_args()

tag = '-a'+str(args.learning_sample)
extra_tag = args.extra_tag
exp = args.experience
if exp=='1d' :
    extra_tag = extra_tag+'-1d'
if exp=='std' :
    extra_tag = extra_tag+'-std'

if tag=='-a1' : learning_sample = 'LHS'
else : learning_sample = 'orbits'

print('\n > Learning sample : %s.'%learning_sample)

# -------------------------------------------------------------------------------- #
# ---------------------    Loading available 'observations'   -------------------- #
# -------------------------------------------------------------------------------- #

# 'Observations' can be loaded from the 'datatset' directory.  
# They are then used to calculate longterm metric (e.g., mean, std, covariance)
xt_truth = np.load('dataset/xt_truth.npz')['arr_0']
yt_truth = np.load('dataset/yt_truth.npz')['arr_0']


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
    if exp=='1d' :
        x_data = np.delete(x_data, (3,4), axis=-1)
        y_data = y_data[:,-1]
        y_data = y_data.reshape(-1,1)

# Loading lhs learning sample
if tag=='-a1' :
    print(' > Loading learning sample of LHS sample.')
    x_data = np.load('dataset/x_data-a1'+extra_tag+'.npz')['arr_0'][0]
    y_data = np.load('dataset/y_data-a1'+extra_tag+'.npz')['arr_0'][0][...,:3]


# --- Learning fhat_betas
print('\n ------ Learning fhat_thetas ------- ')
# Normalization of x & y data
norm_input = False
if norm_input :
    mean_x, std_x = np.mean(x_data, axis=0), np.std(x_data, axis=0)
    mean_y, std_y = np.mean(y_data, axis=0), np.std(y_data, axis=0)
    x_data = (x_data-mean_x)/std_x
    y_data = (y_data-mean_y)/std_y


if exp=='2d' : 
    liste_dims = [0,1,2,4,5]
else :
    liste_dims = [0,1,2,5]
    y_data = y_data[:,-1]
    y_data = y_data.reshape(-1,1)
    if norm_input :
        mean_y, std_y = mean_y[-1], std_y[-1]

x_data = x_data[...,liste_dims]
if norm_input :
    mean_x, std_x = mean_x[liste_dims], std_x[liste_dims]
    print('mean_x : ', mean_x)



# Setting up NN model
if exp=='2d' :
    layers = [1024, 512, 256, 128, 64, 32, 16]
else :
    layers = [256, 128, 64, 32, 16]

   
n_epochs = 25

dic_NN = {'name':'f_orb', 'in_dim':len(liste_dims), 'out_dim':y_data.shape[1], 
        'nlays':layers, 'dropout':False, 'exp':exp}
nn_L63 = ML_model(dic_NN)
if norm_input :
    nn_L63.norms = [mean_x, mean_y, std_x, std_y]
else : 
    extra_tag += '-noNorm'
nn_L63.suffix = tag+extra_tag
print(nn_L63.model.summary())

print(' > Training NN model.')
train_ML_model(x_data, y_data, nn_L63, batch_size=32, n_epochs=n_epochs, 
        split_mode='random', split_ratio=0.15)

print(' > Loading model weights.')
nn_L63.model.load_weights('weights/best-weights'+nn_L63.suffix+'.h5')

print(' > Done.')
exit()
