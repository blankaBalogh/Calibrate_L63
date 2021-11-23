import numpy as np
from pyDOE import lhs

from ML_model import ML_model, train_ML_model
from data import generate_data, generate_LHS, generate_data_solvers, generate_x0 
from metrics import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import tensorflow.keras.backend as K


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
'''

# Setting up script parameters
tag         = '-a2' #'-a'+str(args.learning_sample)
extra_tag   = '' #args.extra_tag
exp         = '1d' #args.experience

if exp == '2d' :
    extra_tag = extra_tag+'-2d'


# -------------------------------------------------------------------------------- #
# ----------------    2nd EXPERIMENT : LEARNING dx = f(x,theta)   ---------------- #
# -------------------------------------------------------------------------------- #

# Getting R2 score for validation.
def r2_score_keras(y_truth, y_pred) : 
    '''
    R2-score using numpy arrays. 
    '''
    import tensorflow.keras.backend as K
    print('y_truth shape : ', y_truth.shape)
    print('y_pred shape : ', y_pred.shape)
    y_truth = tf.convert_to_tensor(y_truth)
    y_pred = tf.convert_to_tensor(y_pred)
    num = K.sum((y_truth - y_pred)**2)
    denom = K.sum((y_truth - K.mean(y_truth, axis=0))**2)
    return (1-num/denom)


'''
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

x_data_, y_data_ = np.copy(x_data), np.copy(y_data)
x_data = x_data.T.reshape(x_data.shape[-1],-1).T
y_data = y_data.T.reshape(y_data.shape[-1],-1).T

# Computing u
K = x_data.shape[-1]-4 
J = int(y_data.shape[-1]/K)

h,c,F,b = x_data[...,-4], x_data[...,-3], x_data[...,-2], x_data[...,-1]
h = np.repeat(h,K).reshape(-1,1)
c = np.repeat(c,K).reshape(-1,1)
F = np.repeat(F,K).reshape(-1,1)
b = np.repeat(b,K).reshape(-1,1)

u_data  = -(h*c/b)*np.sum(y_data.reshape(y_data.shape[0],K,J),axis=-1).reshape(-1,1)
u_data  = u_data.reshape(-1,1)
x_data_x= x_data[...,:K].reshape(-1,1)
x_data_b= b.reshape(-1,1)
x_data = np.concatenate((x_data_x, x_data_b),axis=-1)

#del y_data

# Normalization of x & y data
norm_input = False

if norm_input :
    mean_x, std_x = np.mean(x_data, axis=0), np.std(x_data, axis=0)
    mean_u, std_u = np.mean(u_data, axis=0), np.std(u_data, axis=0)
    x_data = (x_data-mean_x)/std_x
    u_data = (u_data-mean_u)/std_u

x_data_ = np.copy(x_data)
x_data = x_data[...,0].reshape(-1,1)
'''

# Setting up NN model
layers = [32,32]
n_epochs    = 30
dic_NN      = {'name':'f_lhs', 'in_dim':1, 'out_dim':1, 'nlays':layers, 
    'dropout':False, 'exp':exp}

# Loading model 1
nn_L96      = ML_model(dic_NN)

nn_model = nn_L96.model
print(nn_model.summary())


#from sklearn.model_selection import train_test_split
#x_train, x_test, u_train, u_test = train_test_split(x_data, u_data, 
#                test_size=0.15, random_state=42)

#ckpt_10e = ModelCheckpoint('weights/weights-b0.h5', 
#            monitor='val_r2_score_keras', save_weights_only=True, verbose=1, mode="max",
#            period=10)
#ckpt_best = ModelCheckpoint('weights/best-weights-b0.h5', 
#            monitor='val_r2_score_keras', save_best_only=True, verbose=1, mode="max")

learning_rate = 1e-3
K,J = 8,32
loss_fn = tf.keras.losses.MeanSquaredError()        # loss function. Can be custom. 
optim   = tf.keras.optimizers.Adam(learning_rate)

nn_model.compile(loss=loss_fn, optimizer=optim, metrics=[r2_score_keras])

print(' > Loading model weights.')
nn_model_c5 = nn_model
nn_model_c5.load_weights('weights/best-weights-b0_b5.h5')

# Loading model 2
nn_L96_c10  = ML_model(dic_NN)

nn_model_c10 = nn_L96_c10.model
print(nn_model_c10.summary())
nn_model_c10.compile(loss=loss_fn, optimizer=optim, metrics=[r2_score_keras])
nn_model_c10.load_weights('weights/best-weights-b0_b10.h5')

print(' > Done.')



# Plotting
import matplotlib.pyplot as plt

xt_truth_c5 = np.load('data/xt_truth_b5.npz')['arr_0'][:,0]
yt_truth_c5 = np.load('data/yt_truth_b5.npz')['arr_0'][:,0]
xt_truth_c10 = np.load('data/xt_truth_b10.npz')['arr_0'][:,0]
yt_truth_c10 = np.load('data/yt_truth_b10.npz')['arr_0'][:,0]

ysum_c5 = np.sum(yt_truth_c5.reshape(yt_truth_c5.shape[0],K,J),axis=-1)
ysum_c10 = np.sum(yt_truth_c10.reshape(yt_truth_c10.shape[0],K,J),axis=-1)
h,c = 1.,10. 
B_c5 = (-h*c/5.)*ysum_c5
B_c10 = (-h*c/10.)*ysum_c10

Bpred_c10 = nn_model_c10.predict(xt_truth_c10[:,0])
Bpred_c5 = nn_model_c5.predict(xt_truth_c5[:,0])




#xt_truth,yt_truth=np.load('data/xt_truth.npz')['arr_0'],np.load('data/yt_truth.npz')['arr_0']
#xt_truth = xt_truth[:,0]
#ysum = np.sum(yt_truth[:,0].reshape(yt_truth.shape[0],K,J),axis=-1)
#h,c,F,b = xt_truth[0,-4],xt_truth[0,-3],xt_truth[0,-2],xt_truth[0,-1]
#B = (-h*c/b)*ysum[:,0]
#Bpred = nn_model.predict(xt_truth[:,0].reshape(-1,1))

plt.scatter(xt_truth_c5[:,0],B_c5[:,0],label='truth b=5',marker='+',s=1,color='red')
plt.scatter(xt_truth_c10[:,0],B_c10[:,0],label='truth b=10',marker='+',s=1,color='royalblue')
plt.scatter(xt_truth_c5[:,0],Bpred_c5,label='pred b=5',marker='+',s=1,color='darkred')
plt.scatter(xt_truth_c10[:,0],Bpred_c10,label='pred b=10',marker='+',s=1,color='darkblue')
plt.xlabel(r'$X$')
plt.ylabel(r'$B$')
plt.legend()
plt.show()







