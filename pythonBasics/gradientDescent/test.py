import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def peaks(x,y):
    x,y = np.meshgrid(x,y)
    z = 1 - 2*x + 1*y
    return z


def plotGraphWithoutSigmoid():
    x = np.linspace(-4,4,101)
    y = np.linspace(-4,4,101)

    Z = peaks(x,y)
    plt.imshow(Z, vmin=-4, vmax=4, origin='lower')
    # plt.imshow(Z, extent=[x[0], x[-1],y[0], y[-1]], vmin=-5, vmax=5, origin='lower')
    plt.colorbar()
    plt.show()

def plotGraphWitSigmoid():
    x = np.linspace(-4,4,101)
    y = np.linspace(-4,4,101)

    Z = peaks(x,y)
    Z = sigmoid(Z)
    plt.imshow(Z,origin='lower')
    # plt.imshow(Z, extent=[x[0], x[-1],y[0], y[-1]], vmin=-5, vmax=5, origin='lower')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    # plottingTwoDimensionFunction()
    plotGraphWithoutSigmoid()
    plotGraphWitSigmoid()