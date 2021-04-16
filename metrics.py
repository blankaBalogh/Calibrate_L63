import numpy as np
from data import *

# Distances
def dist_L1() :
    '''
    L1 distance.
    '''

    def distance(pred, truth) : 
        '''
        '''
        return abs(pred-truth)

    return distance



def dist_L2() :
    '''
    L2 distance.
    '''

    def distance(pred, truth) :
        '''
        '''
        return np.square(pred-truth)

    return distance


# Loss functions
# Validation orbit generated + loss computed over it
def compute_loss_data(nn_L63, xt_truth, x0=[0.,0.,1.,10.,28.,8/3], 
        n_steps=200, dt=0.05, alpha=1.) :
    '''
    Loss function.
    '''
    n_snapshots = len(x0)
    distance = dist_L2()
    if alpha == 0.5 :
        print(' > metric : mean+STD.')
    elif alpha == 1. :
        print(' > metric : STD only.')

    def loss_fun(theta_to_update) :
        '''
        '''
        thetas = np.repeat(theta_to_update, n_snapshots)
        x0[...,3:] = thetas

        output = generate_data(nn_L63, x0, n_steps=200, dt=dt, compute_y=False)
        xt_pred = output['x']
        
        mean_hat_x = np.mean(xt_pred[...,:3], axis=(0,1))
        mean_truth_x = np.mean(xt_truth, axis=(0))

        std_hat_x = np.std(xt_pred[...,:3], axis=(0,1))
        std_truth_x = np.std(xt_truth, axis=(0))
        
        err_mean = np.mean(distance(mean_hat_x, mean_truth_x))
        err_std = np.mean(distance(std_hat_x, std_truth_x))
        err = alpha*err_std + (1-alpha)*err_mean
        print('error : ', err)
    #print('Optimal beta : ', beta_to_update)
        return err

    return loss_fun



# Loss function for kriging
def compute_loss_kriging(gp) :
    '''
    Loss function to optimize beta value after kriging. 
    '''
    
    def loss_fun(theta_to_update) :
        '''
        '''
        theta_ = np.array([theta_to_update])
        print('Optimal theta value : %f.' % theta_)

        err = gp.predict(theta_) 
        return err[0,0]

    return loss_fun


"""
# callback functions
def generate_callback_fun() :
    '''
    Generates a callback function.
    '''
    global n_iter, steps
    n_iter=1
    step = []

    def callback(x_) :
        '''
        Function to print all intermediate values
        '''
        if n_iter%10 == 0 :
            print("Iteration no.", n_iter)
            print("theta value : ", x_)
        steps.append(x_)
        n_iter += 1

    return callback
"""
