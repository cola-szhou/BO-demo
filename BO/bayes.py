def black_box_function(x, y):
    return -x ** 2 - (y - 1) ** 2 + 1


from bayes_opt import BayesianOptimization, UtilityFunction
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits import mplot3d

np.random.seed(42)
xs = np.linspace(-2, 10, 10000)


def f(x):
    return np.sin(x) + np.cos(0.5 * x)  # np.exp(-(x - 2) ** 2) + np.exp(-(x - 6) ** 2 / 10) + 1 / (x ** 2 + 1) + y


def posterior(optimizer, x_obs, y_obs, grid):
    optimizer._gp.fit(x_obs, y_obs)

    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma


def plot_gp(optimizer, x, y):
    fig = plt.figure(figsize=(16, 10))
    steps = len(optimizer.space)
    fig.suptitle(
        'Bayesian Optimization After {} Steps'.format(steps),
        fontdict={'size': 30}
    )

    gs = gridspec.GridSpec(2, 1)
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])

    x_obs = np.array([[res["params"]["x"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])

    mu, sigma = posterior(optimizer, x_obs, y_obs, x)
    axis.plot(x, y, linewidth=3, label='Target')
    axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label=u'Observations', color='r')
    axis.plot(x, mu, '--', label='Prediction')

    axis.fill(np.concatenate([x, x[::-1]]),
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
              alpha=.1, label='95% confidence interval')

    axis.set_xlim((-2, 10))
    axis.set_ylim((None, None))
    axis.set_ylabel('f(x)')
    axis.set_xlabel('x')

    utility_function = UtilityFunction(kind="ucb", kappa=5, xi=0)
    utility = utility_function.utility(x, optimizer._gp, 0)
    acq.plot(x, utility, label='Utility Function', color='purple')
    acq.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15,
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    acq.set_xlim((-2, 10))
    acq.set_ylim((0, np.max(utility) + 0.5))
    acq.set_ylabel('Utility')
    acq.set_xlabel('x')

    plt.show()


def target(x, y):
    # return np.exp(-(x - 2) ** 2) + np.exp(-(x - 6) ** 2 / 10) + 1 / (x ** 2 + 1)
    return - (100 * np.sqrt(np.abs(y - 0.01 * x ** 2)) + 0.01 * np.abs(x + 10))


# x = np.linspace(-15, -5)
# y = np.linspace(-3, 3)
# X, Y = np.meshgrid(x, y)
# Z = target(X, Y)
#
# plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_wireframe(X, Y, Z)
# plt.show()


x = np.linspace(0, 10)
y = f(x)

optimizer = BayesianOptimization(f, {'x': (0, 10)}, random_state=123)
optimizer.maximize(init_points=2, n_iter=30, kappa=5, acq='ucb')


plot_gp(optimizer, x, y)
for i in range(10):
    optimizer.maximize(init_points=0, n_iter=1, kappa=5, acq='ucb')

    plot_gp(optimizer, x, y)
