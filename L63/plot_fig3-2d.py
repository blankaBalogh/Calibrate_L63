import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap


# Loading results file
results = np.load('results/results-optimalBetas-noStand-alpha1.npz')['arr_0']

# Getting arrays from 'results' file
biased_sigmas, biased_rhos  = results[:,0], results[:,1]
optimal_betas               = results[:,2]
err_betaOpt, err_betaStand  = results[:,3], results[:,4]
diff_errors_1d              = err_betaStand - err_betaOpt

# Formatting data to dataframe to plot heatmap
data        = np.copy(results[:,:4])
data[:,-1]  = diff_errors_1d
df          = pd.DataFrame(data, columns=['sigma', 'rho', 'beta_star', 'diff'])
df          = df.pivot(index='sigma', columns='rho', values=['diff', 'beta_star'])
beta_star   = np.array(df['beta_star'])
df          = df['diff']


# Defining colormap
colors      = cm.get_cmap('Greens', 31)
newcolors   = colors(np.linspace(0, 1, 21))
pink        = np.array([248/256, 24/256, 148/256, 1])
newcmp      = ListedColormap(newcolors)
newcmp.set_under(pink)


# Plotting figure
fig = plt.figure(figsize=(9,7))
ax = sns.heatmap(df, annot=beta_star, fmt='.2f', vmin=0., vmax=0.1, 
        cmap=newcmp, zorder=1, cbar_kws={'extend':'both'})
xlabels = ['{:,.2f}'.format(x) for x in np.linspace(26.5, 29.5, 10)]
ylabels = ['{:,.2f}'.format(x) for x in np.linspace(9.5, 10.5, 10)]
ax.set_xticklabels(xlabels)
ax.set_yticklabels(ylabels)
ax.set_xlabel(r'$\rho$', fontsize=15)
ax.set_ylabel(r'$\sigma$', fontsize=15)
ax.axvline(x=5., ls='--', color='red', zorder=2)
ax.axhline(y=5., ls='--', color='red', zorder=3)
plt.show()
