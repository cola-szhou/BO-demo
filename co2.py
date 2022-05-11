import GPy.kern
import numpy as np
import json
import matplotlib.pyplot as plt
import GPy
from IPython.display import display
from plot import plot

year = []
ppm = []

k = GPy.kern.RBF(input_dim=1) \
    + GPy.kern.sde_StdPeriodic(input_dim=1) * GPy.kern.RBF(input_dim=1) \
    + GPy.kern.sde_RatQuad(input_dim=1) \
    + GPy.kern.sde_Bias(input_dim=1)  # By default, the parameters are set to 1.

fig, ax = plt.subplots(figsize=(8, 6))
k.lengthscale = 1.0
k.plot(ax=ax)


with open("co2-mm-mlo_json.json", 'r', encoding='utf8') as f:
    _d = json.load(f)
    for i in range(len(_d)):
        if _d[i]['Average'] > 0:
            year.append(_d[i]['Decimal Date'])
            ppm.append(_d[i]['Average'])

year = np.array(year)
ppm = np.array(ppm)

index = np.random.randint(0, 720, 500)
x_training = np.array([[year[i]] for i in index])
y_training = np.array([[ppm[i]] for i in index])

x_linspace = np.arange(1958.0, 2018., 0.001)

kernel = GPy.kern.RBF(input_dim=1) \
         + GPy.kern.sde_StdPeriodic(input_dim=1) \
         + GPy.kern.sde_RatQuad(input_dim=1) \
         + GPy.kern.sde_Bias(input_dim=1)

m = GPy.models.GPRegression(x_training, y_training, kernel)
m.log_likelihood()

display(m)

m.optimize(messages=True)
m.optimize_restarts(num_restarts=15)

y_predict, std = m.predict(x_linspace[:, np.newaxis])
y_predict = y_predict[:, 0]
std = std[:, 0]

plot(x_linspace, year, ppm, y_predict, std, x_training, y_training)

display(m)
