import numpy as np
import GPy
from matplotlib import pyplot as plt
from IPython.display import display
from plot import plot

GPy.plotting.change_plotting_library("matplotlib")

x_training = np.random.uniform(-3., 3., (50, 1))
y_training = np.sin(x_training) + np.cos(x_training) + np.random.randn(50, 1) * 0.05
x_linspace = np.arange(-3., 3., 0.001)

# kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
# kernel = GPy.kern.Exponential(input_dim=1, variance=1.,lengthscale=1.)
# kernel =GPy.kern.Matern32(input_dim=1, variance=1., lengthscale=1.)
# kernel =GPy.kern.Matern52(input_dim=1, variance=1., lengthscale=1.)
#kernel = GPy.kern.Brownian(input_dim=1, variance=4.)
kernel = GPy.kern.PeriodicExponential(input_dim=1) * GPy.kern.Brownian(input_dim=1, variance=4.)

m = GPy.models.GPRegression(x_training, y_training, kernel)

display(m)

m.optimize(messages=True)
m.optimize_restarts(num_restarts=2)

y_predict, std = m.predict(x_linspace[:, np.newaxis])
y_predict = y_predict[:, 0]
std = std[:, 0]

plot(x_linspace, y_predict, std, x_training, y_training)

display(m)
"""

x_training = np.random.uniform(-3., 3., (50, 2))
y_training = np.sin(x_training[:, 0:1]) * np.sin(x_training[:, 1:2]) + np.random.randn(50, 1) * 0.05
x_1 = np.arange(-3, 3, 1)[0]
x_linspace = np.array((0.1, np.arange(-3, 3, 1)[i]) for i in range(3))

print(x_linspace)
kernel = GPy.kern.Matern52(2, ARD=True) + GPy.kern.White(2)

m = GPy.models.GPRegression(x_training, y_training, kernel)
m.optimize(messages=True, max_f_eval=1000)
m.optimize_restarts(num_restarts=10)

y_predict, std = m.predict(x_linspace)
plt = m.plot()
"""
