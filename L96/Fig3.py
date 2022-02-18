import numpy as np
import matplotlib.pyplot as plt
import GPy
from scipy.optimize import minimize

def loss_kriging(m) :
    def loss_fun(th) :
        theta_ = np.array([th])
        err = m.predict(theta_)[0][:,0]
        return err[0]
    return loss_fun



F_values = np.array([9.0,9.25,9.5,9.75,10.0,10.25,10.5,10.75,11.])
cstar_values = np.zeros(len(F_values))
err_cstar = np.zeros(len(F_values))
err_c0 = np.zeros(len(F_values))
err_lr = np.zeros(len(F_values))

# Truth dataset
spinup = 300
K,J = 8,32
xt_truth = np.load('xt_truth.npz')['arr_0'][spinup:,0,:K]
std_truth = np.mean(np.std(xt_truth,axis=0),axis=-1)

cstar_mean = np.array([7.7931,8.3134,8.8541,9.4221,10.0063,10.6174,11.2377,11.8782,
    12.5463])


err_cstar_2000 = np.load('errors_cstar_2000.npz')['arr_0']
err_cstar_3000 = np.load('errors_cstar_3000.npz')['arr_0']
err_cstar_5000 = np.load('errors_cstar_5000.npz')['arr_0']

err_c0_2000 = np.load('errors_c0_2000.npz')['arr_0']
err_c0_3000 = np.load('errors_c0_3000.npz')['arr_0']
err_c0_5000 = np.load('errors_c0_5000.npz')['arr_0']

err_lr = np.load('errors_lr.npz')['arr_0']

err_c0 = np.concatenate((err_c0_2000,err_c0_3000,err_c0_5000),axis=0)
err_cstar = np.concatenate((err_cstar_2000,err_cstar_3000,err_cstar_5000),axis=0)

log_errors = True
if log_errors :
    err_c0 = np.log(err_c0)
    err_cstar = np.log(err_cstar)
    err_lr = np.log(err_lr)

sample_size = err_c0.shape[0]
err_c0_mean = np.mean(err_c0,axis=0)
err_c0_std = np.std(err_c0,axis=0)
err_cstar_mean = np.mean(err_cstar,axis=0)
err_cstar_std = np.std(err_cstar,axis=0)


# Computing confidence intervals at 95%.
zval = 1.96

#ic_c0=zval*c0_std/np.sqrt(sample_size)
#ic_cstar=zval*cstar_std/np.sqrt(sample_size)
ic_err_c0=zval*err_c0_std/np.sqrt(sample_size)
ic_err_cstar=zval*err_cstar_std/np.sqrt(sample_size)

c0s = np.repeat(10., len(F_values))
width = np.min(np.diff(F_values))/3


fig, ax = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [3,1]}, 
        figsize=(7,7))

# Top panel
ax[0].plot(F_values, err_cstar_mean, color='mediumblue', marker='o', 
label=r'optimal',lw=.5)
ax[0].fill_between(F_values,(err_cstar_mean-ic_err_cstar),(err_cstar_mean+ic_err_cstar),
        color='mediumblue',alpha=0.5)
ax[0].plot(F_values, err_c0_mean, color='k', marker='o',label=r'standard',lw=.5)
ax[0].fill_between(F_values,(err_c0_mean-ic_err_c0),(err_c0_mean+ic_err_c0),
        color='k',alpha=0.5)
ax[0].plot(F_values, err_lr, color='red', marker='o', label=r'linear regression',
        lw=.5)
ax[0].axvline(10.,ls='--',color='k',label=r'$c_0$',lw=1)
#ax[0].axhline(0.,ls='--',color='k',lw=1)
#ax[0].set_xlabel(r'$F_b$')
ax[0].set_ylabel(r'log(MSE)')
ax[0].legend()
#ax[0].set_ylim(-13.,0.)

plt.subplots_adjust(hspace=0.)

# Bottom panel

ax[1].bar(F_values-width, cstar_mean, width, color='mediumblue', align='edge')
ax[1].bar(F_values, c0s, width, color='k', align='edge')
ax[1].yaxis.set_label_position('right')
ax[1].yaxis.tick_right()
ax[1].set_xlabel(r'$F_b$')
ax[1].set_ylabel(r'$c$', fontsize=13)
ax[1].set_ylim(0,13)

ax[1].axvline(10.,color='red',lw=1.5,ls='--')
ax[1].axhline(10.,color='red',lw=1.5,ls='--')
ax[1].text(8.85,10.65,r'$c=c_0$',color='red')
ax[1].text(10.02,10.7,r'$F_c=F_0$',color='red')



plt.savefig('Fig3.pdf',format='pdf')
#plt.show()


