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
'''

# Setting up script parameters
tag         = '-a2' #'-a'+str(args.learning_sample)
extra_tag   = ''    #args.extra_tag
exp         = '1d'  #args.experience

if exp == '2d' :
    extra_tag = extra_tag+'-2d'


if tag == '-a1' : learning_sample = 'LHS'
else : learning_sample = 'orbits'

print('\n > Learning sample : %s.'%learning_sample)


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

x_data, y_data = x_data[25:], y_data[25:]
x_data_, y_data_ = np.copy(x_data), np.copy(y_data)
x_data = x_data.T.reshape(x_data.shape[-1],-1).T
y_data = y_data.T.reshape(y_data.shape[-1],-1).T

# Computing u
K = x_data.shape[-1]-4 
J = int(y_data.shape[-1]/K)

#h,c,F,b = x_data[...,-4], x_data[...,-3], x_data[...,-2], x_data[...,-1]
#h = np.repeat(h,K).reshape(-1,1)
#c = np.repeat(c,K).reshape(-1,1)
#F = np.repeat(F,K).reshape(-1,1)
#b = np.repeat(b,K).reshape(-1,1)
h,c,F,b = x_data[0,-4],x_data[0,-3],x_data[0,-2],x_data[0,-1]
print('c : ', c)
print('F : ', F)

u_data  = -(h*c/b)*np.sum(y_data.reshape(y_data.shape[0],K,J),axis=-1).reshape(-1,1)
u_data  = u_data.reshape(-1,1)
x_data_x= x_data[...,:K].reshape(-1,1)
x_data_c= np.repeat(x_data[...,-3],K).reshape(-1,1)
print('x data b : ', x_data_b.shape)
print('x data x : ', x_data_x.shape)
x_data = np.concatenate((x_data_x, x_data_c),axis=-1)

#del y_data

# Normalization of x & y data
norm_input = False

if norm_input :
    mean_x, std_x = np.mean(x_data, axis=0), np.std(x_data, axis=0)
    mean_u, std_u = np.mean(u_data, axis=0), np.std(u_data, axis=0)
    x_data = (x_data-mean_x)/std_x
    u_data = (u_data-mean_u)/std_u

#x_data_ = np.copy(x_data)
#x_data = x_data[...,0].reshape(-1,1)

# Setting up NN model
layers = [32,32]
n_epochs    = 15
dic_NN      = {'name':'f_lhs', 'in_dim':x_data.shape[-1], 'out_dim':1, 'nlays':layers, 
    'dropout':False, 'exp':exp}
nn_L96      = ML_model(dic_NN)

nn_model = nn_L96.model
print(nn_model.summary())


from sklearn.model_selection import train_test_split
x_train, x_test, u_train, u_test = train_test_split(x_data, u_data, 
                test_size=0.15, random_state=42)

ckpt_10e = ModelCheckpoint('weights/weights-b0'+extra_tag+'.h5', 
            monitor='val_r2_score_keras', save_weights_only=True, verbose=1, mode="max",
            period=10)
ckpt_best = ModelCheckpoint('weights/best-weights-b0'+extra_tag+'.h5', 
            monitor='val_r2_score_keras', save_best_only=True, verbose=1, mode="max")

learning_rate = 1e-3
loss_fn = tf.keras.losses.MeanSquaredError()        # loss function. Can be custom. 
optim   = tf.keras.optimizers.Adam(learning_rate)

nn_model.compile(loss=loss_fn, optimizer=optim, metrics=[r2_score_keras])


print(' > Training NN model.')
valid_data = ([x_test[...,0], x_test[...,1]], u_test)
history = nn_model.fit([x_train[...,0], x_train[...,1]], u_train, epochs=n_epochs,
        batch_size=32, verbose=0, validation_data=valid_data, 
        callbacks=[ckpt_10e, ckpt_best])
#train_ML_model(x_data, u_data, nn_L96, batch_size=32, n_epochs=n_epochs, 
#        split_mode='random', split_ratio=0.15)


print(' > Loading model weights.')
nn_model.load_weights('weights/best-weights-b0'+extra_tag+'.h5')

"""
# Plotting results
import matplotlib.pyplot as plt
xt_truth = np.load('data/xt_truth'+extra_tag+'.npz')['arr_0'][:,0]
yt_truth = np.load('data/yt_truth'+extra_tag+'.npz')['arr_0'][:,0]

h,c = 1.,10.
ysum = np.sum(yt_truth.reshape(yt_truth.shape[0],K,J),axis=-1)
B_truth = (-(h*c/b)*ysum).reshape(-1,1)
print('B_truth shape : ', B_truth.shape)
print('xt_truth shape : ', xt_truth.shape)
xt_truth = xt_truth[...,:K].reshape(-1,1)
b = np.repeat(10., xt_truth.shape[0]).reshape(-1,1)
B_pred = nn_model.predict([xt_truth,b])
B_ptrain = nn_model.predict([x_data[:,0], x_data[:,1]])

fig, ax = plt.subplots(ncols=2)

ax[0].scatter(x_data[:,0], u_data, label='truth', marker='+', s=1)
ax[0].scatter(x_data[:,0], B_ptrain, label='pred', marker='+', s=1)
plt.legend()
ax[0].set_xlabel(r'$X$')
ax[0].set_ylabel(r'$B$')
ax[0].set_title('train')

ax[1].scatter(xt_truth, B_truth, label='truth',marker='+', s=1)
ax[1].scatter(xt_truth, B_pred, label='pred', marker='+', s=1)
plt.legend()
ax[1].set_xlabel(r'$X$')
ax[1].set_ylabel(r'$B$')
ax[1].set_title('truth')

plt.show()



"""


