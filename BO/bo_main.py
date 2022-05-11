import GPy
import GPy.kern
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy import optimize

# 最小化问题

from test_functions import test_function


def PI(x, gp, y_max, xi):
    mean, std = gp.predict(x)
    z = (mean - y_max - xi) / std
    unit_norm = stats.norm()
    return unit_norm.cdf(z)


def EI(x, gp, y_min, xi):
    mean, std = gp.predict(x[:, np.newaxis])
    std = std ** 0.5
    std[std == 0] = np.inf
    delta = (y_min - mean - xi)

    z = delta / std
    unit_norm = stats.norm()
    return delta * unit_norm.cdf(z) + std * unit_norm.pdf(z)


def UCB(x, gp, kappa):
    mean, std = gp.predict(x[:, np.newaxis])
    return mean + kappa * std


def acq_max(x, ac, gp, y_min, bounds, n_warmup=45):
    x0 = np.random.uniform(low=bounds[0], high=bounds[1], size=(n_warmup, 1))
    proposal = None
    best_ei = -np.inf
    for x0_ in x0:
        ei = np.max(ac(x, gp, y_min, x0_))
        if ei > best_ei:
            proposal = x0_
    return proposal


def plot(iteration, x, mu, var, x_true, y_true):
    plt.suptitle(f"Iteration {iteration}")
    plt.title("$f(x^*)$")
    plt.plot(x, mu, color="b", label='predict')
    c = 1.96
    plt.fill_between(x, mu - c * var, mu + c * var, alpha=.15, fc='b', ec='None', label='std')
    plt.plot(x_true, y_true, 'r.', label='true_data')
    plt.plot(x, test_function(x), 'g', label="target")
    plt.legend()
    plt.show()


# 主函数
def main():
    iteration = 0
    n_iter = 30
    x_training = np.random.uniform(0, 1, (2, 1))
    y_training = test_function(x_training)
    x_linspace = np.arange(0, 1, 0.01)
    bounds = [0, 1]

    # kernel = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=1)
    kernel = GPy.kern.Matern52(input_dim=1)
    m = GPy.models.GPRegression(x_training, y_training, kernel)
    m.optimize()
    # m.optimize_restarts(num_restarts=10)

    mu, var = m.predict(x_linspace[:, None])
    mu = mu[:, 0]
    var = var[:, 0]
    plt.figure()
    plt.plot(x_linspace, test_function(x_linspace))
    plt.show()

    plt.figure()
    plt.plot(x_training, y_training, 'r.')
    plt.plot(x_linspace, test_function(x_linspace),'g-')
    plt.plot(x_linspace, mu, 'b-')
    c = 1.96
    plt.fill_between(x_linspace, mu - c * var, mu + c * var, alpha=.25, fc='b', ec='None', label='std')
    plt.show()

    while iteration < n_iter:
        # 根据acquisition函数计算下一个试验点
        suggestion = acq_max(
            x=x_linspace,
            ac=EI,
            gp=m,
            y_min=y_training.min(),
            bounds=bounds,
        )

        # 进行试验（采样），更新观测点集合
        x_training = np.append(x_training, suggestion)
        y_training = np.append(y_training, test_function(suggestion))

        m = GPy.models.GPRegression(x_training[:, np.newaxis], y_training[:, np.newaxis], kernel)
        m.optimize()
        if iteration % 1 == 0:
            mu, var = m.predict(x_linspace[:, None])
            mu = mu[:, 0]
            var = var[:, 0]
            plot(iteration, x_linspace, mu, var, x_training, y_training)
        iteration += 1

    print(y_training.min())

if __name__ == '__main__':
    main()
