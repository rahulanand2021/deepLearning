import numpy as np
import matplotlib.pyplot as plt


def fx(x):
    return 3*x**2 -3*x + 4

def derivativeOf(x):
    return 6*x -3

def adamOptimizerDynamicLRBasedOnGradient():
    x = np.linspace(-3,3,2001)
    localMinimum = np.random.choice(x,1)
    learning_rate = 0.03
    training_epocs = 100
    print(f"Initial Local Minimum (Guess) : {localMinimum}")
    
    modelparams = np.zeros((training_epocs,3))

    for index in range(training_epocs):
        grad = derivativeOf(localMinimum)
        lr = learning_rate*np.abs(grad)
        localMinimum = localMinimum - grad*lr
        modelparams[index,0] = localMinimum[0]
        modelparams[index,1] = grad[0]
        modelparams[index,2] = lr
    print(f"After GD, Actual LocalMinimum : {localMinimum}")

    # plot the gradient over iterations

    fig,ax = plt.subplots(1,3,figsize=(10,3))

    # generate the plots
    for i in range(3):
      ax[i].plot(modelparams[:,i],'o-',markerfacecolor='w')
      ax[i].set_xlabel('Iteration')

    ax[0].set_ylabel('Local minimum')
    ax[1].set_ylabel('Derivative')
    ax[2].set_ylabel('Learning rate')
    ax[2].legend(['Fixed l.r.','Grad-based l.r.','Time-based l.r.'])

    plt.tight_layout()
    plt.show()

def learningDecayDynamicLRBasedOnTime():
    x = np.linspace(-3,3,2001)
    localMinimum = np.random.choice(x,1)
    learning_rate = 0.03
    training_epocs = 100
    print(f"Initial Local Minimum (Guess) : {localMinimum}")
    
    modelparams = np.zeros((training_epocs,3))

    for index in range(training_epocs):
        grad = derivativeOf(localMinimum)
        lr = learning_rate*(1-(index+1)/training_epocs) 
        localMinimum = localMinimum - grad*lr
        modelparams[index,0] = localMinimum[0]
        modelparams[index,1] = grad[0]
        modelparams[index,2] = lr
    print(f"After GD, Actual LocalMinimum : {localMinimum}")

    # plot the gradient over iterations

    fig,ax = plt.subplots(1,3,figsize=(10,3))

    # generate the plots
    for i in range(3):
      ax[i].plot(modelparams[:,i],'o-',markerfacecolor='w')
      ax[i].set_xlabel('Iteration')

    ax[0].set_ylabel('Local minimum')
    ax[1].set_ylabel('Derivative')
    ax[2].set_ylabel('Learning rate')
    ax[2].legend(['Fixed l.r.','Grad-based l.r.','Time-based l.r.'])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # plottingFunctionAndDerivatives()
    # oneDimensionGradientDescent()
    # adamOptimizerDynamicLRBasedOnGradient()
    learningDecayDynamicLRBasedOnTime()