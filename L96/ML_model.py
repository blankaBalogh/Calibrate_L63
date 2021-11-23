'''

Neural Network Model

'''

import numpy as np
import tensorflow as tf
from tensorflow import keras

import tensorflow.keras.backend as K
from tensorflow.keras import layers
import tensorflow.keras.optimizers as ko
import tensorflow.keras.losses as kl

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler


class ML_model():
    '''
    This class defines the ML model.
    Attributes :
        - name      : Name of the NN model.
        - dropout   : If True, adds Dropout layers after each Dense layer.
        - in_dim    : Dimension of input array.
        - out_dim   : Dimension of output array.
        - nlays     : List of nodes on each Dense layers, from the input to the output.
        - norms     : If input normalization : 
                        [mean_input, mean_output, std_input, std_output]
        - suffix    : Name of the model, for saving best weights.
        - exp       : Experience type : '1d' or '2d'.
        - model     : Model defined with input parameters. Available after execution
                      of self.build_model().

    Methods : 
        - set_params(dic)   : Sets init parameters.
        - build_model()     : Builds the model with respect to the input parameters.
        - f()               : Function generating time derivatives to integrate. 
    '''

    def __init__(self, dic=None):

        self.name = ''
        
        self.dropout    = False
        self.in_dim     = 2
        self.out_dim    = 1
        self.nlays      = [64, 32, 16]
        self.norms      = None
        self.suffix     = ''
        self.exp        = '1d'
        self.model      = None

        self.set_params(dic)
        self.build_model()


    def set_params(self, dic):
        if dic is not None:
            for key, val in zip(dic.keys(), dic.values()):
                if key in self.__dict__.keys():
                    self.__dict__[key] = val


    def build_model(self):
        '''
        Builds the NN model with respect to the input parametrs.
        '''

        # Setting activation function to Relu
        activation = 'elu'
        
        # Input layer for state variable x
        inp1_ = tf.keras.Input(shape=1, name='X_data')

        # If L63 parameters are also in the input, input layer for theta
        if not (self.in_dim-1)==0 :
            inp2_ = tf.keras.Input(shape=1, name='Theta')
            inp_ = layers.Concatenate()([inp1_, inp2_])
        else : 
            inp_ = inp1_
        
        # Definition of the NN with respect to model attributes
        if self.dropout==True :
            x = layers.Dropout(0.2)(inp_)
        else :
            x = inp_
        x = layers.Dense(self.nlays[0], activation=activation)(x)
        for k in range(1, len(self.nlays)):
            x = layers.Dense(self.nlays[k], activation=activation)(x)
            
        out_ = layers.Dense(self.out_dim, name='predictions')(x)
         
        if not (self.in_dim-1)==0 :
            model = tf.keras.Model(inputs=[inp1_, inp2_], outputs=out_)
        else :
            model = tf.keras.Model(inputs=inp1_, outputs=out_)
        
        # Compiling model with MSE loss and Adam optimizer
        model.compile(loss="mse", metrics=[r2_score_keras], optimizer="adam")

        # Defining self.model
        self.model = model


    def f(self):
        '''
        Time derivative that will be integrated when generating validation orbits.
        '''

        def func(x) :
            '''
            '''
            if self.norms is not None :
                raw_x, x = np.copy(x), x
                x = (x-self.norms[0])/self.norms[2]
            
            x_, theta_ = x[...,:1], x[...,-1]

            ishp1 = x.shape
            
            # Reformatting x if it has not been formatted for the NN model
            if len(ishp1)==1 :
                x_ = np.reshape(x_, (-1,1))
                theta_ = np.reshape(theta_, (-1,1))

            # 2-D experiment : the NN replaces the whole L63 model
            u_ = self.model.predict_on_batch([x_,theta_])
                
            if self.norms is not None:
                u__ = self.norms[3]*u_ + self.norms[1]
            else : 
                u__ = u_
                      
            return u__
        
        return func





# Other NN-related functions

# Fitting the NN model to the learning sample
def train_ML_model(x_data, u_data, NN_model, batch_size=32, learning_rate=0.001, 
        n_epochs=10, return_datasets=False, split_mode='beta', split_ratio=0.15) :
    '''
    Trains the NN model with respect to the learning sample : (x_data, y_data).
    '''
    x_data_ = np.copy(x_data)
    del x_data
    x_data, th_data = x_data_[...,:1], x_data_[...,-1]
    
    # Train/test split mode 1 : for random datasets (e.g., LHS learning sample).
    #                           The N last valus of the LS are saved for validation.

    if split_mode=='beta' :
        print(' > learning sample : betas.')
        ind_last = int(x_data.shape[0]*split_ratio)
        x_train, x_test     = x_data[:-ind_last], x_data[-ind_last:]
        u_train, u_test     = u_data[:-ind_last], u_data[-ind_last:]
        th_train, th_test   = th_data[:-ind_last], th_data[-ind_last:]
   
    # Train/test split mode 2 : for timeseries. Part of an orbit is set aside as 
    #                           a validation dataset.
    elif split_mode=='timeseries' :
        print(' > learning sample : timeseries.')
        n_ic = (np.unique(th_data)).shape[0]
        n_ts = int(th_data.shape[0]/n_ic)
        ind_last = int(n_ts*split_ratio)
        x_data_ = x_data.reshape(n_ic, n_ts, 1), 
        th_data_= th_data.reshape(n_ic, n_ts, NN_model.in_dim-1)
        u_data_ = u_data.reshape(n_ic, n_ts, NN_model.out_dim)
        x_train, x_test = x_data_[:,:-ind_last], x_data_[:,-ind_last:]
        u_train, u_test = u_data_[:,:-ind_last], u_data_[:,-ind_last:]

        n_trts, n_tets  = int(n_ts*(1-split_ratio)), int(n_ts*split_ratio)
        x_train = x_train.reshape(n_ic*n_trts, 1)
        x_test  = x_test.reshape(n_tets*n_ic, 1)
        u_train = u_train.reshape(n_ic*n_trts, NN_model.out_dim)
        u_test  = u_test.reshape(n_ic*n_tets, NN_model.out_dim)

        th_train, th_test   = th_data_[:,:-ind_last], th_data_[:,-ind_last:]
        th_train, th_test   = th_train.reshape(n_ic*n_trts,1), th_test.reshape(n_ic*n_tets,1)
    
    # Train/test split mode 3 : for random datasets. 'train_test_split' for sklearn.
    elif split_mode=='random' :
        from sklearn.model_selection import train_test_split
        print(' > learning sample : random.')
        nf_x, nf_u = x_data_.shape[1], u_data.shape[1]
        print('   nf_x : ', nf_x)
        train_data = np.concatenate((x_data_, u_data), axis=-1)
        np.random.seed(42)
        np.random.shuffle(train_data)
        x_data, u_data = train_data[...,:nf_x], train_data[...,nf_x:]
        print('   x_data shape : ', x_data.shape)
        
        x_train, x_test, u_train, u_test = train_test_split(x_data, u_data, 
                test_size=split_ratio, random_state=42)
        x_train, th_train = x_train[...,:1], x_train[...,-1]
        x_test, th_test = x_test[...,:1], x_test[...,-1]
    
    # Model checkpoint : saving best model weights wrt "monitor" score.
    # To monitor the fit of the model, weights are also saved every 10 epochs.
    n_train = x_train.shape[0]
    n_batch_per_epoch = int(n_train/batch_size)
    n_save_epochs = 10
    ckpt_10e = ModelCheckpoint('weights/weights'+NN_model.suffix+'-e{epoch:02d}.h5', 
            monitor='val_r2_score_keras', save_weights_only=True, verbose=1, mode="max",
            period=10)
    ckpt_best = ModelCheckpoint('weights/best-weights'+NN_model.suffix+'.h5', 
            monitor='val_r2_score_keras', save_best_only=True, verbose=1, mode="max")

    # Learning rate scheduler (if needed).
    def scheduler(epoch, lr):
        if epoch < 15:
            return lr
        else:
            return lr * np.exp(-0.05)

    # Fitting NN model
    loss_fn = tf.keras.losses.MeanSquaredError()        # loss function. Can be custom. 
    optim   = tf.keras.optimizers.Adam(learning_rate)   # optimizer

    def get_lr_metric(optimizer):
        def lr(y_true, y_pred):
            return optimizer._decayed_lr(tf.float32)    # I use ._decayed_lr method instead of .lr
        return lr
    lr_metric = get_lr_metric(optim)

    # Validation R2 metric is defined below.
    NN_model.model.compile(loss=loss_fn, optimizer=optim, metrics=[r2_score_keras, 
        lr_metric])

    print('training %s model...'%NN_model.name)
    inputs = x_train

    inputs = [x_train, th_train]
    valid_data = ([x_test, th_test], u_test)

    history = NN_model.model.fit(inputs, u_train, epochs=n_epochs,
        batch_size=batch_size, verbose=0, validation_data=valid_data, 
        callbacks=[ckpt_10e, ckpt_best, LearningRateScheduler(scheduler)])
    
    # Loading best ANN weights
    NN_model.model.load_weights('weights/best-weights'+NN_model.suffix+'.h5')
    
    if return_datasets :
        return x_train, x_test, th_train, th_test, u_train, u_test


# Getting R2 score for validation.
def r2_score_keras(y_truth, y_pred) : 
    '''
    R2-score using numpy arrays. 
    '''
    num = K.sum((y_truth - y_pred)**2)
    denom = K.sum((y_truth - K.mean(y_truth, axis=0))**2)
    return (1-num/denom)

