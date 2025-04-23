import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def softMaxUsingNumPy(z):
    if z is None:
        z = [1,2,3]
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    sigma = numerator/denominator
    print(f" Numerator is {numerator}")
    print(f" Denominator is {denominator}")
    print(f" Sigma is {sigma}")
    print(np.sum(sigma))
    plotGraph(z, sigma,isPytorch=False)

def softMaxUsingNumPyWithRandom():
    z = np.random.randint(1,10,20)
    softMaxUsingNumPy(z)

def plotGraph(z, sigma, isPytorch):
    plt.plot(z, sigma, 'ko')
    plt.xlabel('Original Number (z)')
    plt.ylabel('Sigma Value $\sigma$')
    # plt.yscale('log')   Optional if we want to see the log scale in the y axis
    if(isPytorch) :
        plt.title('$\sum\sigma$ = %g' %sigma.sum())
    else:
        plt.title('$\sum\sigma$ = %g' %np.sum(sigma))
    plt.show()

def softMaxUsingPytorchWithRandom():
    z = np.random.randint(1,10,20)
    softFunction = nn.Softmax(dim=0)
    sigmaT = softFunction(torch.Tensor(z))
    print(sigmaT)
    plotGraph(z,sigmaT, isPytorch=True)

def plotNaturalExponent_e():
    z= np.random.randint(1,100000,2000)
    eValue = np.exp(z)
    #logValues = np.log(eValue)  # this should equal z
    logValues = np.log(z)
    plt.plot(z, eValue, 'ko-', label='exp(z)')
    #plt.plot(z, logValues, 'r^-', label='log(exp(z))')
    plt.plot(z, logValues, 'r^-', label='log((z))')

    plt.xlabel('Original Number (z)')
    plt.ylabel('Value')
    plt.title('Natural Exponent and Log Values')
    plt.legend()
    plt.grid(True)
    plt.show()

def plotLogValues():
    z = np.linspace(0.001, 5, 200)
    logValue = np.log(z)
    plt.plot(z, logValue, 'ko')
    plt.xlabel('Original Number (z)')
    plt.ylabel('Log Value $\log$')
    plt.show()



if __name__ == "__main__" :
    #plotNaturalExponent_e()
    # softMaxUsingNumPy(None)
    #softMaxUsingNumPyWithRandom()
    #softMaxUsingPytorchWithRandom()
    plotLogValues()
   