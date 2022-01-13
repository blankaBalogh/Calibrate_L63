import numpy as np
import matplotlib.pyplot as plt

# Loading results
res = np.load('results/results_evalRhos-stdOnly.npz')['arr_0']

rhos, betaStar = res[:,0], res[:,1]
err_opt, err_std = res[:,2], res[:,3]

betas = np.repeat(8/3., betaStar.shape[0])
width = np.min(np.diff(rhos))/3   


# plotting layout
fig, ax = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [3,1]}, 
        figsize=(7,7))


ax[0].plot(rhos, err_opt, color='mediumblue', label='opt')
ax[0].plot(rhos, err_std, color='black', label='stand')
ax[0].axhline(0., color='k', lw=1)
ax[0].set_ylabel(r'$m$', fontsize=13)
ax[0].legend()

plt.subplots_adjust(hspace=0.)

ax[1].bar(rhos-width, betaStar, width, color='mediumblue', align='edge')
ax[1].bar(rhos, betas, width, color='k', align='edge')
ax[1].text(29.2,2.72,r'$\beta_0=8/3$',color='red')
ax[1].axhline(8/3., color='red', ls='--')

ax[1].yaxis.set_label_position('right')
ax[1].yaxis.tick_right()
ax[1].set_ylim(2.3, 3.)
ax[1].set_xlabel(r'$\rho$', fontsize=13)
ax[1].set_ylabel(r'$\beta$', fontsize=13)

plt.show()

