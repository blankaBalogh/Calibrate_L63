import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Loading datasets
thetas = np.load('train_thetas_gp-a2-m2.npz')['arr_0']
errors = np.load('train_errors_gp-a2-m2.npz')['arr_0']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

p = ax.scatter(thetas[:,0], thetas[:,1], thetas[:,2], c=errors)

ax.set_xlabel('sigma')
ax.set_ylabel('rho')
ax.set_zlabel('beta')

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')
ax.grid(False)


plt.colorbar(p)

plt.show()


