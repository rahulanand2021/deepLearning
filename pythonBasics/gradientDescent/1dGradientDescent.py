import numpy as np
import matplotlib.pyplot as plt


def fx(x):
    return 3*x**2 -3*x + 4

def derivativeOf(x):
    return 6*x -3

def oneDimensionGradientDescent():
    x = np.linspace(-3,3,2001)
    plt.plot(x,fx(x), x, derivativeOf(x))
    plt.xlim(x[[0,-1]])
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend(['y', 'dy'])
    plt.show()

if __name__ == "__main__":
    oneDimensionGradientDescent()