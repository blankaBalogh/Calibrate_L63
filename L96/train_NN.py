import numpy as np
from pyDOE import lhs

from ML_model import ML_model, train_ML_model
from data import generate_data, generate_LHS, generate_data_solvers, generate_x0 
from metrics import *

# Sharing available GPU resources
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus :
    tf.config.experimental.set_memory_growth(gpu, True)


# Parsing aruments
from argparse import ArgumentParser
parser = ArgumentParser()

# Experience type (a=1 : LHS, a=2 : orbit)
parser.add_argument('-a', '--learning_sample', default=1,
        help='Learning sample selection : orbit (=2) or lhs (=1).')
# Specific name of the dataset
parser.add_argument('-et', '--extra_tag', type=str, default='', 
        help='Adds an extra tag. Useful to save new datasets.')
# Experience type : '1d' or '2d'
parser.add_argument('-exp', '--experience', type=str, default='1d',
        help="Experience type : '2d' or '1d'.")

args    = parser.parse_args()

# Setting up script parameters
tag         = '-a'+str(args.learning_sample)
extra_tag   = args.extra_tag
exp         = args.experience

if exp == '1d' :
    extra_tag = extra_tag+'-1d'

if tag == '-a1' : learning_sample = 'LHS'
else : learning_sample = 'orbits'
print('\n > Learning sample : %s.'%learning_sample)


# -------------------------------------------------------------------------------- #
# ----------------    2nd EXPERIMENT : LEARNING dx = f(x,theta)   ---------------- #
# -------------------------------------------------------------------------------- #
print(' > Loading learning sample. ')

# Loading 'orbits' learning sample
if tag=='-a2' :
    print(' > Loading learning sample of orbits.')
    x_data  = np.load('data/x_data-a2'+extra_tag+'.npz')['arr_0']
    y_data  = np.load('data/y_data-a2'+extra_tag+'.npz')['arr_0']
    #x_data, y_data = np.swapaxes(x_data,0,1), np.swapaxes(y_data,0,1) 
    #x_data = x_data.reshape(-1, x_data.shape[-1])
    #y_data = y_data.reshape(-1, y_data.shape[-1])


# Loading lhs learning sample
if tag=='-a1' :
    print(' > Loading LHS learning sample.')
    x_data = np.load('data/x_data-a1'+extra_tag+'.npz')['arr_0'][0]
    y_data = np.load('data/y_data-a1'+extra_tag+'.npz')['arr_0'][0]

# Computing u
K = x_data.shape[-1]-4 
J = int(y_data.shape[-1]/K)

h,c,F,b = x_data[...,-4], x_data[...,-3], x_data[...,-2], x_data[...,-1]
h = np.repeat(h,K).reshape(h.shape[0],K)
c = np.repeat(c,K).reshape(c.shape[0],K)
F = np.repeat(F,K).reshape(F.shape[0],K)
b = np.repeat(b,K).reshape(b.shape[0],K)

u_data  = -h*c*(b**(-1))*np.sum(y_data.reshape(y_data.shape[0],K,J),axis=-1)
u_data  = u_data.reshape(-1,1)
x_data_x= x_data[...,:K].reshape(-1,1)
x_data_b= b.reshape(-1,1)
x_data = np.concatenate((x_data_x, x_data_b),axis=-1)

del y_data

# Normalization of x & y data
norm_input = True

if norm_input :
    mean_x, std_x = np.mean(x_data, axis=0), np.std(x_data, axis=0)
    mean_u, std_u = np.mean(u_data, axis=0), np.std(u_data, axis=0)
    x_data = (x_data-mean_x)/std_x
    u_data = (u_data-mean_u)/std_u


# Setting up NN model
layers = [32, 16]
n_epochs    = 3
dic_NN      = {'name':'f_lhs', 'in_dim':2, 'out_dim':1, 'nlays':layers, 
    'dropout':False, 'exp':exp}
nn_L96      = ML_model(dic_NN)

if norm_input :
    nn_L96.norms = [mean_x, mean_u, std_x, std_u]

else : 
    extra_tag += '-noNorm'

nn_L96.suffix = tag+extra_tag

print(nn_L96.model.summary())

print(' > Training NN model.')
train_ML_model(x_data, u_data, nn_L96, batch_size=32, n_epochs=n_epochs, 
        split_mode='random', split_ratio=0.15)

print(' > Loading model weights.')
nn_L96.model.load_weights('weights/best-weights'+nn_L96.suffix+'.h5')

print(' > Done.')
exit()
