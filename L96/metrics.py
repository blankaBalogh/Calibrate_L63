import numpy as np
from data import *



# Loss functions
# Validation orbit generated + loss computed over it
def compute_loss_data(nn_L63, xt_truth, x0=[0.,0.,1.,10.,28.,8/3], 
        n_steps=20000, dt=0.05, alpha=1., tag='', extra_tag='') :
    '''
    Loss function.
    '''
    n_snapshots = len(x0)
    distance = dist_L2()
    if alpha == 0.5 :
        print(' > metric : mean+STD.')
    elif alpha == 1. :
        print(' > metric : STD only.')

    def loss_fun(theta_to_update, i=0) :
        '''
        '''
        thetas = np.repeat(theta_to_update, n_snapshots).reshape(3, n_snapshots).T
        x0[...,3:] = thetas
        output = generate_data(nn_L63, x0, n_steps=n_steps, dt=dt, compute_y=False)
        xt_pred = output['x']

        save_ds=False
        if save_ds :
            np.savez_compressed('dataset/saved_xt/saved_orbits/xt_pred_'+\
                    str(i)+'-numpy.npz', xt_pred)

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
def compute_loss_kriging(gp, norm_gp=False, norms=None) :
    '''
    Loss function to optimize beta value after kriging. 
    '''
    
    def loss_fun(th) :
        '''
        '''
        #theta_to_update = np.zeros(3)
        #theta_to_update[0], theta_to_update[1] = 10., 28.
        #theta_to_update[2] = th
        #theta_ = np.copy(theta_to_update) 
        theta_ = np.array([th])
        print('Optimal theta value : ', theta_)
        
        if norm_gp :
            theta_ = (theta_-norms[0])/norms[2]

        err = gp.predict(theta_)[0][:,0]

        if norm_gp :
            err = norms[3]*err + norms[1]

        print('step error : %.4f.'%err)
        return err[0]

    return loss_fun


# Loss function for kriging
def compute_STDloss_kriging(gp, std_truth, norm_gp=False, norms=None) :
    '''
    Loss function to optimize beta value after kriging. 
    '''
    
    def loss_fun(theta_to_update) :
        '''
        '''
        theta_ = np.array([theta_to_update])
        print('Optimal theta value : ', theta_)
        
        if norm_gp :
            theta_ = (theta_-norms[0])/norms[2]
    
        std_pred = gp.predict(theta_)[0][0]
        
        if norm_gp :
            std_pred = norms[3]*std_pred + norms[1]

        err = np.mean((std_pred-std_truth)**2)

        return err

    return loss_fun


