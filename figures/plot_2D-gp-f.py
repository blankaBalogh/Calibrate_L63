import numpy as np
import matplotlib.pyplot as plt
import GPy

fdir = '../dataset/'
xt_truth = np.load(fdir+'xt_truth.npz')['arr_0']

# Kriging learning sample
train_errors = np.load(fdir+'020621/train_errors_gp-a1-small2.npz')['arr_0']
train_thetas = np.load(fdir+'020621/train_thetas_gp-a1-small2.npz')['arr_0']

# Fitting kriging model
kernel = GPy.kern.Matern52(input_dim=3, ARD=True)

m = GPy.models.GPRegression(train_thetas, train_errors, kernel)
m.optimize(messages=True)
print(m)


# Predicting Errors
x, y = np.linspace(26.5,29.5,10), np.linspace(2.,3.,10) #np.linspace(9.,11.,10) 
sigmas_rhos = np.array(np.meshgrid(x, y)).T.reshape(-1,2)
z = np.repeat(10., 100)
Thetas_to_predict = np.zeros((100,3))
Thetas_to_predict[:,0] = z #sigmas_rhos[:,0]
Thetas_to_predict[:,1] = sigmas_rhos[:,0]
Thetas_to_predict[:,2] = sigmas_rhos[:,1]

predicted_errors = m.predict(Thetas_to_predict)[0][:,0]

Errors = np.zeros((100,3))
Errors[:,:2] = sigmas_rhos
Errors[:,2] = predicted_errors

Errors = np.swapaxes(Errors,0,1).reshape(3,10,10).T
errors = Errors[:,:,2]


# Plotting figure
fig = plt.figure(figsize=(10,8))

plt.pcolormesh(x,y,errors)#, vmin=0., vmax=1.)
plt.axvline(28., color='red', ls='--')
plt.axhline(8/3, color='red', ls='--')
plt.xlabel('rho')
plt.ylabel('beta')
plt.title('errors with sigma=10.')

plt.colorbar(extend='both')

#plt.show()
plt.savefig('plots/rho_beta_sigma10_kriging_fhat_std.png')


