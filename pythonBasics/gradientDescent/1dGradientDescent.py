import numpy as np
import matplotlib.pyplot as plt


def fx(x):
    return 3*x**2 -3*x + 4

def derivativeOf(x):
    return 6*x -3

def plottingFunctionAndDerivatives():
    x = np.linspace(-3,3,2001)
    plt.plot(x,fx(x), x, derivativeOf(x))
    plt.xlim(x[[0,-1]])
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend(['y', 'dy'])
    plt.show()

def oneDimensionGradientDescent():
    x = np.linspace(-3,3,2001)
    # np.set_printoptions(threshold=np.inf)   -- This is used if you want to see all the values of x
    localMinimum = np.random.choice(x,1)
    learning_rate = 0.03
    training_epocs = 100
    print(f"Initial Local Minimum (Guess) : {localMinimum}")
    for index in range(training_epocs):
        grad = derivativeOf(localMinimum)
        localMinimum = localMinimum - grad*learning_rate
    print(f"After GD, Actual LocalMinimum : {localMinimum}")

def oneDimensionGradientDescentToPlotGraph():
    x = np.linspace(-3,3,2001)
    localMinimum = np.random.choice(x,1)
    learning_rate = 0.03
    training_epocs = 100
    print(f"Initial Local Minimum (Guess) : {localMinimum}")
    
    modelparams = np.zeros((training_epocs,2))
    for index in range(training_epocs):
        grad = derivativeOf(localMinimum)
        localMinimum = localMinimum - grad*learning_rate
        modelparams[index,0] = localMinimum[0]
        modelparams[index,1] = grad[0]
    print(f"After GD, Actual LocalMinimum : {localMinimum}")
    # print(modelparams)

    # plot the gradient over iterations

    fig,ax = plt.subplots(1,2,figsize=(12,4))

    for i in range(2):
        ax[i].plot(modelparams[:,i],'o-')
        ax[i].set_xlabel('Iteration')
        ax[i].set_title(f'Final estimated minimum: {localMinimum[0]:.5f}')

        ax[0].set_ylabel('Local minimum')
        ax[1].set_ylabel('Derivative')

    plt.show()
if __name__ == "__main__":
    # plottingFunctionAndDerivatives()
    # oneDimensionGradientDescent()
    oneDimensionGradientDescentToPlotGraph()