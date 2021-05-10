import numpy as np
import matplotlib.pyplot as plt

fdir = '../dataset/'
xt_truth = np.load(fdir+'xt_truth.npz')['arr_0']

# Kriging learning sample
train_errors = np.load(fdir+'thetas_errors/train_errors_gp-a1-m1.npz')['arr_0']
train_thetas = np.load(fdir+'thetas_errors/train_thetas_gp-a1-m1.npz')['arr_0']

# Fitting GP Regressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
kernel = Matern(length_scale=1, nu=2.5)
gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
train_thetas = train_thetas.reshape(-1,3)
gp.fit(train_thetas, train_errors)

# Predicting Errors
x, y = np.linspace(26.5,29.5,10), np.linspace(2.,3.,10) # np.linspace(9.,11.,10),
sigmas_rhos = np.array(np.meshgrid(x, y)).T.reshape(-1,2)
z = np.repeat(10., 100)
Thetas_to_predict = np.zeros((100,3))
Thetas_to_predict[:,0] = z
Thetas_to_predict[:,1] = sigmas_rhos[:,0]
Thetas_to_predict[:,2] = sigmas_rhos[:,1]

predicted_errors = gp.predict(Thetas_to_predict)[:,0]

Errors = np.zeros((100,3))
Errors[:,:2] = sigmas_rhos
Errors[:,2] = predicted_errors

Errors = np.swapaxes(Errors,0,1).reshape(3,10,10).T
errors = Errors[:,:,2]


# Plotting figure
fig = plt.figure(figsize=(6,5))

plt.pcolormesh(x,y,errors, vmin=0., vmax=1.)
plt.axvline(28., color='red', ls='--')
plt.axhline(8/3, color='red', ls='--')
plt.xlabel('rho')
plt.ylabel('beta')
plt.title('errors with sigma=10.')

plt.colorbar(extend='both')

plt.show()


