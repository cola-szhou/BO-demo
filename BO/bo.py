import GPy
import GPyOpt
from numpy.random import seed
import matplotlib.pyplot as plt
import numpy as np

from GPyOpt.util.general import reshape


def grlee12(x):
    return np.sin(10 * np.pi * x) / (2 * x) + np.power(x - 1, 4)


def bukin6(x):
    x = reshape(x, 2)
    x1 = x[:, 0]
    x2 = x[:, 1]
    return 100 * np.sqrt(np.abs(x2 - 0.01 * x1 ** 2)) + 0.01 * np.abs(x1 + 10)


def boha1(x):
    x = reshape(x, 2)
    x1 = x[:, 0]
    x2 = x[:, 1]
    return x1 ** 2 + 2 * x2 ** 2 - 0.3 * np.cos(3 * np.pi * x1) - 0.4 * np.cos(4 * np.pi * x2) + 0.7


def f(x):
    return np.power(k*x-2,2)*np.sin(12*x-4) + np.random.normal(scale=0.2, size=len(x))


def plot(bo, f_true):
    plt.figure(figsize=(8, 5))
    plt.subplot(2, 1, 1)
    x = np.linspace(0, 1)
    plt.plot(bo.x_opt, bo.fx_opt, 'ko', markersize=10, label='Best found')
    plt.plot(bo.X, bo.Y, 'k.', markersize=8, label='Observed stations')
    #plt.plot(x, , label='True')

    x_grid = np.arange(0, 1, 0.001)
    x_grid = x_grid.reshape(len(x_grid), 1)
    m, v = bo.model.predict(x_grid.reshape(len(x_grid), 1))
    print(x_grid)

    acqu = bo.acquisition.acquisition_function(x_grid)
    acqu_normalized = (-acqu - min(-acqu)) / (max(-acqu - min(-acqu)))

    plt.plot(x_grid, m, label='mean')
    plt.plot(x_grid, m - 1.96 * np.sqrt(v), 'k-', alpha=0.2)
    plt.plot(x_grid, m + 1.96 * np.sqrt(v), 'k-', alpha=0.2)
    plt.legend()
    plt.subplot(2, 1, 2)
    factor = max(m + 1.96 * np.sqrt(v)) - min(m - 1.96 * np.sqrt(v))
    # plt.plot(x_grid, 0.2 * factor * acqu_normalized - abs(min(m - 1.96 * np.sqrt(v))) - 0.25 * factor, 'r-', lw=2,
    #         label='Acquisition (arbitrary units)')
    plt.plot(x_grid, acqu_normalized, 'r-', lw=2, label='Acquisition')
    plt.legend()
    plt.show()


f_true = GPyOpt.objective_examples.experiments1d.forrester()
f_sim = GPyOpt.objective_examples.experiments1d.forrester(sd=5)

bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (0, 1)}]


objective = GPyOpt.core.task.SingleObjective(f)
feasible_region = GPyOpt.Design_space(space=bounds)

initial_design = GPyOpt.experiment_design.initial_design('random', feasible_region, 5)
k = GPy.kern.RBF(input_dim=1)
model = GPyOpt.models.GPModel(exact_feval=True, optimize_restarts=10, verbose=False, kernel=k, noise_var=0.2)
aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(feasible_region)
acquisition = GPyOpt.acquisitions.AcquisitionEI(model, feasible_region, optimizer=aquisition_optimizer)
evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

bo = GPyOpt.methods.ModularBayesianOptimization(model, feasible_region, objective, acquisition, evaluator,
                                                initial_design)


# bo = GPyOpt.methods.BayesianOptimization(f=f,  # function to optimize
#                                          domain=bounds,  # box-constraints of the problem
#                                          acquisition_type='EI',
#                                          exact_feval=True)  # Selects the Expected improvement
max_iter = 10  # maximum time 40 iterations
max_time = 60  # maximum time 60 seconds

bo.run_optimization(max_iter, max_time, verbosity=False)

plot(bo, f_true)

# myBopt2D.plot_acquisition()

bo.plot_acquisition()
# bo.plot_convergence()
print(bo.x_opt)
print(bo.fx_opt)
