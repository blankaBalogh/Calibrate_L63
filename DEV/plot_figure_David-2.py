import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

# Loading results file
results = np.load('results-linspace-STDOnly-newNN-kriging-1ts.npz')['arr_0']
#results = results[:-1]
biased_sigmas, biased_rhos = results[:,0], results[:,1]
optimal_betas = results[:,2]
err_betaOpt, err_betaStand = results[:,3], results[:,4]

"""
# histogram of optimal beta values
plt.hist(optimal_betas)
plt.axvline(8/3, color='k', ls='--', label='truth beta')
plt.xlabel('optimal beta')
plt.title('Optimal beta values repartition', fontsize=15, y=1.05)
plt.ylabel('counts')
plt.show()
"""


# raster plot of errors
diff_errors_1d = err_betaStand - err_betaOpt
diff_errors = diff_errors_1d.reshape(10,10)
sigmas = biased_sigmas.reshape(10,10)
rhos = biased_rhos.reshape(10,10)
#sigmas = np.linspace(9.5, 10.5, 10)
#rhos = np.linspace(26.5, 29.5, 10)
blues = cm.get_cmap('Blues', 31)
newcolors = blues(np.linspace(0, 1, 21))
pink = np.array([248/256, 24/256, 148/256, 1])
newcolors[:2, :] = pink
newcmp = ListedColormap(newcolors)


"""
fig, ax = plt.subplots()
plot = sns.heatmap(sigmas, rhos, diff_errors,  
        vmin=-0.01, vmax=0.1, cmap=newcmp)
plt.axvline(10, color='k', ls='--', label='truth sigma')
plt.axhline(28, color='k', ls='--', label='truth rho')
plt.xlabel('sigma')
plt.ylabel('rho')
plt.title('relative error : err(beta*) - err_standard', fontsize=15, y=1.05)
plt.colorbar(plot, extend='both')
plt.show()
"""

blues = cm.get_cmap('Greens', 31)
newcolors = blues(np.linspace(0, 1, 21))
pink = np.array([248/256, 24/256, 148/256, 1])
newcolors[:2, :] = pink
newcmp = ListedColormap(newcolors)

data = np.copy(results[:,:4])
data[:,-1] = diff_errors_1d
df = pd.DataFrame(data, columns=['sigma', 'rho', 'beta_star', 'diff'])
df = df.pivot(index='sigma', columns='rho', values=['diff', 'beta_star'])
beta_star = np.array(df['beta_star'])
df = df['diff']

"""
# raster plot of errors
diff_errors_1d = err_betaStand - err_betaOpt
diff_errors = diff_errors_1d.reshape(10,10)
sigmas = biased_sigmas.reshape(10,10)
rhos = biased_rhos.reshape(10,10)
#sigmas = np.linspace(9.5, 10.5, 10)
#rhos = np.linspace(26.5, 29.5, 10)
"""

print(np.max(diff_errors_1d))
fig = plt.figure(figsize=(9,7))
ax = sns.heatmap(df, annot=beta_star, fmt='.2f', vmin=-0.01, vmax=0.1, cmap=newcmp, zorder=1)
xlabels = ['{:,.2f}'.format(x) for x in np.linspace(26.5, 29.5, 10)]
ylabels = ['{:,.2f}'.format(x) for x in np.linspace(9.5, 10.5, 10)]
ax.set_xticklabels(xlabels)
ax.set_yticklabels(ylabels)
ax.axvline(x=5., ls='--', color='red', zorder=2)
ax.axhline(y=5., ls='--', color='red', zorder=3)
#ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.set_title('relative error : err_standard - err(beta*)', fontsize=15, y=1.05)
plt.savefig('/Users/blankab/Desktop/figure_erreurs_v2-largerNN-kriging-1ts.png')
plt.show()


"""
# Older version
fig, ax = plt.subplots()
plot = ax.pcolormesh(sigmas, rhos, diff_errors,  vmin=-0.01, vmax=0.1, cmap=newcmp)
plt.axvline(10, color='k', ls='--', label='truth sigma')
plt.axhline(28, color='k', ls='--', label='truth rho')
plt.xlabel('sigma')
plt.ylabel('rho')
plt.title('relative error : err(beta*) - err_standard', fontsize=15, y=1.05)
plt.colorbar(plot, extend='both')
plt.show()
"""

