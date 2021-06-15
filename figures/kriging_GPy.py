import numpy as np
import matplotlib.pyplot as plt
import GPy

fdir = '../dataset/'
xt_truth = np.load(fdir+'xt_truth.npz')['arr_0']

# Kriging learning sample
train_errors = np.load(fdir+'thetas_errors/train_errors_gp-f-STD.npz')['arr_0']
train_thetas = np.load(fdir+'thetas_errors/train_thetas_gp-f-STD.npz')['arr_0']

train_x, train_y = train_thetas[:-25], train_errors[:-25]
valid_x, valid_y = train_thetas[-25:], train_errors[-25:]

# Fitting kriging model
kernel = GPy.kern.Matern52(input_dim=3)

m = GPy.models.GPRegression(train_x, train_y, kernel)
m.optimize(messages=True)
print(m)

pred = m.predict(valid_x)
pred_y, pred_var = pred[0], pred[1]




