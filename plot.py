import matplotlib.pyplot as plt
import numpy as np


def plot(x_linspace, x_true, y_true, y_predict, std, x_training, y_training):
    fig = plt.figure()
    plt.plot(x_true, y_true, color='green', label='true_data')
    # plt.plot(x_linspace, np.sin(x_linspace) + np.cos(x_linspace), 'r:', label=u'$f(x) = sin(x)$')
    plt.plot(x_linspace, y_predict, 'b-', label='predict')
    c = 1.96  # for confidence interval
    plt.fill_between(x_linspace, y_predict - c * std, y_predict + c * std, alpha=.5, fc='b', ec='None', label='std')
    # plt.plot(x_training, y_training, 'r.', markersize=10, label='true')
    plt.ylabel('$f(x)$')
    plt.xlabel('$x$')
    # fig.legend(bbox_to_anchor=(0.45, 0.88), fancybox=True, shadow=True)
    plt.savefig('fig_3.pdf', bbox_inches="tight", dpi=300)
    plt.show()
