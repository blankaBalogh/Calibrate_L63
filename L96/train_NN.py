import numpy as np
from pyDOE import lhs

from ML_model import ML_model, train_ML_model
from data import generate_data, generate_LHS, generate_data_solvers, generate_x0 
from metrics import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import tensorflow.keras.backend as Kb


# Sharing available GPU resources
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus :
    tf.config.experimental.set_memory_growth(gpu, True)

'''
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
tag         = '-a2' #'-a'+str(args.learning_sample)
extra_tag   = ''    #args.extra_tag
exp         = '1d'  #args.experience

if exp == '2d' :
    extra_tag = extra_tag+'-2d'


if tag == '-a1' : learning_sample = 'LHS'
else : learning_sample = 'orbits'

print('\n > Learning sample : %s.'%learning_sample)
'''
K,J = 8,32
# -------------------------------------------------------------------------------- #
# ----------------    2nd EXPERIMENT : LEARNING dx = f(x,theta)   ---------------- #
# -------------------------------------------------------------------------------- #

# Getting R2 score for validation.
def r2_score_keras(y_truth, y_pred) : 
    '''
    R2-score using numpy arrays. 
    '''
    import tensorflow.keras.backend as Kb
    y_truth = tf.convert_to_tensor(y_truth)
    y_pred = tf.convert_to_tensor(y_pred)
    num = Kb.sum((y_truth - y_pred)**2)
    denom = Kb.sum((y_truth - Kb.mean(y_truth, axis=0))**2)
    return (1-num/denom)


# Loading learning sample
spinup = 300

x = np.load('data/x_data-a2.npz')['arr_0']
y = np.load('data/y_data-a2.npz')['arr_0']
h,F,b = x[0,0,-4], x[0,0,-2], x[0,0,-1]
c,x = x[:,:,-3], x[:,:,:K]


# Reshaping learning sample : (N,k+4) -> (N*k,2)
carr = np.zeros((c.shape[0],c.shape[1],K))
for i in range(K) :
    carr[:,:,i] = c

B = -h/b*carr*np.sum(y.reshape(y.shape[0],y.shape[1],K,J),axis=-1)

x = x[spinup:].reshape(-1,1)
carr = carr[spinup:].reshape(-1,1)
B = B[spinup:].reshape(-1,1)

x = np.concatenate((x,carr),axis=-1)

x_data_, u_data_ = np.copy(x), np.copy(B)


# Setting up NN model
layers = [32,32]
n_epochs    = 15
dic_NN      = {'name':'f_lhs', 'in_dim':x_data_.shape[-1], 'out_dim':1, 'nlays':layers, 
    'dropout':False, 'exp':'1d'}
nn_L96      = ML_model(dic_NN)

nn_model = nn_L96.model
print(nn_model.summary())

# Splitting learning sample into train/test samples
from sklearn.model_selection import train_test_split
x_train, x_test, u_train, u_test = train_test_split(x_data_, u_data_, 
                test_size=0.15, random_state=42)

ckpt_best = ModelCheckpoint('weights/best-weights-full.h5', 
            monitor='val_r2_score_keras', save_best_only=True, verbose=1, mode="max")

learning_rate = 1e-3
loss_fn = tf.keras.losses.MeanSquaredError()        # loss function. Can be custom. 
optim   = tf.keras.optimizers.Adam(learning_rate)

nn_model.compile(loss=loss_fn, optimizer=optim, metrics=[r2_score_keras])


print(' > Training NN model.')
valid_data = ([x_test[...,0], x_test[...,1]], u_test)
history = nn_model.fit([x_train[...,0], x_train[...,1]], u_train, epochs=n_epochs,
        batch_size=32, verbose=0, validation_data=valid_data, 
        callbacks=[ckpt_best])

# Loading weights of the best model
print(' > Loading model weights.')
extra_tag = '-largerNN'
nn_L96.model.load_weights('weights/best-weights-full.h5')


