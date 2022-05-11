import math
import numpy as np
import matplotlib.pyplot as plt


# GRAMACY & LEE (2012) FUNCTION http://www.sfu.ca/~ssurjano/grlee12.html
def test_function(x):
    # y = np.sin(10 * np.pi * x) / (2 * x) + np.power(x - 1, 4)

    # y = np.sin(10 * np.pi * x) / (2 * x) + np.power(x - 1, 4)

    y = np.power(6 * x - 2, 2) * np.sin(12 * x - 4)
    return y


if __name__ == '__main__':
    x = np.arange(0.5, 2.5, 0.01)
    y = np.sin(10 * np.pi * x) / (2 * x) + np.power(x - 1, 4)
    plt.plot(x, y)
    plt.show()
