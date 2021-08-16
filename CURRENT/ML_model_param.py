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
    '''

    def __init__(self, dic=None):

        self.name = ''
        
        self.dropout    = False
        self.in_dim     = 6
        self.out_dim    = 3
        self.nlays      = [64, 32, 16]
        self.norms      = None
        self.suffix     = ''
        self.exp        = '2d'
        self.model      = None
        self.sigma      = 10.
        self.rho        = 28.
        self.beta       = 8/3

        self.set_params(dic)
        self.build_model()


    def set_params(self, dic):
        if dic is not None:
            for key, val in zip(dic.keys(), dic.values()):
                if key in self.__dict__.keys():
                    self.__dict__[key] = val


    def build_model(self):
        '''
        '''
        activation = 'relu'
         
        inp1_ = tf.keras.Input(shape=3, name='X_data')
        if not (self.in_dim-3)==0 :
            inp2_ = tf.keras.Input(shape=self.in_dim-3, name='Theta')
            inp_ = layers.Concatenate()([inp1_, inp2_])
        else : 
            inp_ = inp1_
        
        if self.dropout==True :
            x = layers.Dropout(0.2)(inp_)
        else :
            x = inp_
        x = layers.Dense(self.nlays[0], activation=activation)(x)
        for k in range(1, len(self.nlays)):
            x = layers.Dense(self.nlays[k], activation=activation)(x)
            
        out_ = layers.Dense(self.out_dim, name='predictions')(x)
        
        if not (self.in_dim-3)==0 :
            model = tf.keras.Model(inputs=[inp1_, inp2_], outputs=out_)
        else :
            model = tf.keras.Model(inputs=inp1_, outputs=out_)
        # model.summary()
        model.compile(loss="mse", metrics=[r2_score_keras], optimizer="adam")

        self.model = model


    def f(self):
        '''
        '''
        def func(x) :
            if self.norms is not None :
                len_input = len(self.norms[0])
                raw_x, x = np.copy(x), x[:,:len_input]
                x = (x-self.norms[0])/self.norms[2]
        
            x_, theta_ = x[:, :3], x[:,3:]

            ishp1 = x.shape

            if len(ishp1)==1 :
                x_ = np.reshape(x_, (-1, 3))
                theta_ = np.reshape(theta_, (-1, (self.in_dim-3)))

            if self.out_dim == 3 : 
                if not (self.in_dim-3)==0 :
                    y_ = self.model.predict_on_batch([x_, theta_])
                else :
                    y_ = self.model.predict_on_batch(x_)

                if self.norms is not None:
                    y__ = self.norms[3]*y_ + self.norms[1]
                else : 
                    y__ = y_


            elif self.out_dim == 1 :
                sigma, rho = self.sigma, self.rho

                y_ = np.zeros(x.shape)[...,:3]
                y_[..., 0] = sigma * (raw_x[:, 1] - raw_x[:, 0])
                y_[..., 1] = rho*raw_x[:, 0] - raw_x[:, 1] - raw_x[:, 0]*raw_x[:, 2]
                
                if self.norms is not None :
                    if not (self.in_dim-3) == 0 :
                        pred = self.model.predict_on_batch([x_, theta_])
                    else :
                        pred = self.model.predict_on_batch(x_)

                    if len(pred.shape) == 2 :
                        pred = pred[:,0]
                    y_[..., 2] = pred
                    y_[..., 2] = self.norms[3]*y_[..., 2] + self.norms[1]

                else :
                    if not (self.in_dim-3) == 0 :
                        y_[..., 2] = self.model.predict_on_batch([x, theta_])
                    else :
                        y_[..., 2] = self.model.predict_on_batch(x)

                y__ = y_
                      
            return y__
            #return np.reshape(y__, ishp1)
        
        return func


def train_ML_model(x_data, y_data, NN_model, batch_size=512, learning_rate=0.001, 
        n_epochs=10, return_datasets=False, split_mode='beta', split_ratio=0.15) :
    '''
    '''
    x_data_ = np.copy(x_data)
    if not (NN_model.in_dim-3)==0 :
        del x_data
        x_data, th_data = x_data_[:,:3], x_data_[:,3:]
    
    if split_mode=='beta' :
        print(' > learning sample : betas.')
        ind_last = int(x_data.shape[0]*split_ratio)
        x_train, x_test     = x_data[:-ind_last], x_data[-ind_last:]
        y_train, y_test     = y_data[:-ind_last], y_data[-ind_last:]
        
        if not (NN_model.in_dim-3)==0 :
            th_train, th_test   = th_data[:-ind_last], th_data[-ind_last:]
    
    elif split_mode=='timeseries' :
        print(' > learning sample : timeseries.')
        if not (NN_model.in_dim-3)==0 :
            n_ic = (np.unique(th_data)).shape[0]
            n_ts = int(th_data.shape[0]/n_ic)
            ind_last = int(n_ts*split_ratio)
            x_data_, th_data_ = x_data.reshape(n_ic, n_ts, 3), th_data.reshape(n_ic, n_ts, 1)
            y_data_ = y_data.reshape(n_ic, n_ts, 3)
            x_train, x_test = x_data_[:,:-ind_last], x_data_[:,-ind_last:]
            y_train, y_test = y_data_[:,:-ind_last], y_data_[:,-ind_last:]

            n_trts, n_tets = int(n_ts*(1-split_ratio)), int(n_ts*split_ratio)
            x_train, x_test     = x_train.reshape(n_ic*n_trts,3), x_test.reshape(n_tets*n_ic,3)
            y_train, y_test     = y_train.reshape(n_ic*n_trts,3), y_test.reshape(n_ic*n_tets,3)

            th_train, th_test = th_data_[:,:-ind_last], th_data_[:,-ind_last:]
            th_train, th_test   = th_train.reshape(n_ic*n_trts,1), th_test.reshape(n_ic*n_tets,1)
    elif split_mode=='random' :
        from sklearn.model_selection import train_test_split
        print(' > learning sample : random.')
        nf_x, nf_y = x_data_.shape[1], y_data.shape[1]
        print('   nf_x : ', nf_x)
        train_data = np.concatenate((x_data_, y_data), axis=-1)
        np.random.seed(42)
        np.random.shuffle(train_data)
        x_data, y_data = train_data[:,:nf_x], train_data[:,nf_x:]
        print('   x_data shape : ', x_data.shape)
        
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, 
                test_size=split_ratio, random_state=42)
        x_train, th_train = x_train[:,:3], x_train[:,3:]
        x_test, th_test = x_test[:,:3], x_test[:,3:]

    # -- Model checkpoint : saving best model weights wrt "monitor" score.
    n_train = x_train.shape[0]
    n_batch_per_epoch = int(n_train/batch_size)
    n_save_epochs = 10
    ckpt_10e = ModelCheckpoint('weights/weights'+NN_model.suffix+'-e{epoch:02d}.h5', 
            monitor='val_r2_score_keras', save_weights_only=True, verbose=1, mode="max",
            period=10)
            #save_freq=(n_save_epochs*n_batch_per_epoch))
    ckpt_best = ModelCheckpoint('weights/best-weights'+NN_model.suffix+'.h5', 
            monitor='val_r2_score_keras', save_best_only=True, verbose=1, mode="max")


    def scheduler(epoch, lr):
        if epoch < 15:
            return lr
        else:
            return lr * np.exp(-0.05)

    # -- Fitting model
    loss_fn = tf.keras.losses.MeanSquaredError()        # loss function. Can be custom. 
    optim   = tf.keras.optimizers.Adam(learning_rate)   # optimizer

    def get_lr_metric(optimizer):
        def lr(y_true, y_pred):
            return optimizer._decayed_lr(tf.float32) # I use ._decayed_lr method instead of .lr
        return lr
    lr_metric = get_lr_metric(optim)

    NN_model.model.compile(loss=loss_fn, optimizer=optim, metrics=[r2_score_keras, 
        lr_metric])

    print('training %s model...'%NN_model.name)
    inputs = x_train

    valid_data = (x_test, y_test)
    if not (NN_model.in_dim-3) == 0 :
        inputs = [x_train, th_train]
        valid_data = ([x_test, th_test], y_test)

    history = NN_model.model.fit(inputs, y_train, epochs=n_epochs,
        batch_size=batch_size, verbose=0, validation_data=valid_data, 
        callbacks=[ckpt_10e, ckpt_best, LearningRateScheduler(scheduler)])
    
    # -- Loading best ANN weights
    NN_model.model.load_weights('weights/best-weights'+NN_model.suffix+'.h5')
    
    if return_datasets :
        if not (NN_model.in_dim-3)==0 :
            return x_train, x_test, th_train, th_test, y_train, y_test
        else :
            return x_train, x_test, y_train, y_test



# -- Getting 1-R2 score (= Keras R2)
def r2_score_keras(y_truth, y_pred) : 
    '''
    R2-score using numpy arrays. 
    '''
    num = K.sum((y_truth - y_pred)**2)
    denom = K.sum((y_truth - K.mean(y_truth, axis=0))**2)

    return (1-num/denom)


# -- Getting loss function
def get_loss_fn(argument) :
    '''
    Return loss function. 
    '''
    if argument in ["MSE", "mse", "Mse", "MeanSquaredError"] :
        loss = kl.MeanSquaredError()
    elif argument in ["MAE", "mae", "Mae", "MeanAbsoluteError"] :
        loss = kl.MeanAbsoluteError()
    else : 
        return("You asked for optimizer : ", argument, \
                ". Available loss functions are : MSE, MAE.")
    return loss


# -- Getting optimizer
def get_optimizer(argument, lrate, **kwargs) :
    '''
    Returns an optimizer.
    '''
    if argument in ["ADAM", "adam", "Adam"] :
        optim = ko.Adam(lrate)
    elif argument in ["rmsprop", "Rmsprop", "RMSPROP"] :
        optim = ko.RMSprop(lrate)
    elif argument in ["sgd", "SGD"] :
        optim = ko.SGD(lrate)
    else :
        return("You asked for optimizer : ", argument, \
                ". Available loss functions are : MSE, MAE.")
    return optim
