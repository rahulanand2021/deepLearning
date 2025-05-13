import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

x = sym.symbols('x')
fx_expr = sym.sin(x) * sym.exp(-x**2*0.05)
dfx_expr = sym.diff(fx_expr, x)

fx_numeric = sym.lambdify(x, fx_expr, 'numpy')
dfx_numeric = sym.lambdify(x, dfx_expr, 'numpy')

def fx(x):
    return np.sin(x) * np.exp(-x**2*0.5)

def simpleGradientDescent():
    x = np.linspace(-2*np.pi , 2*np.pi , 401)
    # np.set_printoptions(threshold=np.inf)   -- This is used if you want to see all the values of x
    localMinimum = np.random.choice(x,1)
    startingPoint = localMinimum
    learning_rate = 0.01
    training_epocs = 1000
    print(f"Initial Local Minimum (Guess) : {localMinimum}")
    for index in range(training_epocs):
        grad = dfx_numeric(localMinimum)
        localMinimum = localMinimum - grad*learning_rate
    print(f"After GD, Actual LocalMinimum : {localMinimum}")

    plt.plot(x, fx_numeric(x), x,dfx_numeric(x), '--' )
    plt.plot(localMinimum, dfx_numeric(localMinimum) ,  'ro')
    plt.plot(localMinimum, fx_numeric(localMinimum) ,  'ro')
    plt.xlim(x[[0,-1]])
    plt.grid()
    plt.legend(['f(x)', 'df(x)' , 'f(x) minimum'])
    plt.title('Empirical Local Minimum : %s'  %localMinimum)
    plt.figtext(0.5, 0.01, "Starting Point : %s"  %startingPoint, ha="center", fontsize=10)
    plt.show()

def variousStartingLocationwithGradientDescent():
    startLocations = np.linspace(-5,5,500)
    finalRes  = np.zeros(len(startLocations))

    learning_rate = 0.01
    training_epocs = 1000

    for idx, localMinimum in enumerate(startLocations) :

        for index in range(training_epocs):
            grad = dfx_numeric(localMinimum)
            localMinimum = localMinimum - grad*learning_rate
        # print(f"After GD, Actual LocalMinimum : {localMinimum}")
        finalRes[idx] = localMinimum
    
    plt.plot(startLocations, finalRes, 's--')
    plt.xlabel('Starting Guess')
    plt.ylabel('Final Local Minimum after GD')
    plt.show()

def varyingLearningRateWithGradientDescent():
    learningRates = np.linspace(1e-10,1e-1,500)
    finalRes  = np.zeros(len(learningRates))

    training_epocs = 1000

    for idx, learning_Rates in enumerate(learningRates) :
        # Fixed this factor and systematically vary Learning Rates
        localMinimum = 0
        for index in range(training_epocs):
            grad = dfx_numeric(localMinimum)
            localMinimum = localMinimum - grad*learning_Rates
        # print(f"After GD, Actual LocalMinimum : {localMinimum}")
        finalRes[idx] = localMinimum
    
    plt.plot(learningRates, finalRes, 's--')
    plt.grid()
    plt.xlabel('Learning Rates ')
    plt.ylabel('Final Local Minimum after GD')
    plt.show()

def varyingLearningRatesAndTrainingEpocsWithGradientDescent():

    learningRates = np.linspace(1e-10,1e-1,50)
    finalRes  = np.zeros(len(learningRates))

    training_epocs = 1000

    for idx, learning_Rates in enumerate(learningRates) :
        # Fixed this factor and systematically vary Learning Rates
        localMinimum = 0
        for index in range(training_epocs):
            grad = dfx_numeric(localMinimum)
            localMinimum = localMinimum - grad*learning_Rates
        # print(f"After GD, Actual LocalMinimum : {localMinimum}")
        finalRes[idx] = localMinimum
    
    plt.plot(learningRates, finalRes, 's--')
    plt.grid()
    plt.xlabel('Learning Rates ')
    plt.ylabel('Final Local Minimum after GD')
    plt.show()

def plotGraph():
    x = np.linspace(-2*np.pi , 2*np.pi , 401)
    plt.plot(x, fx_numeric(x), x,dfx_numeric(x) )
    plt.grid()
    plt.legend(['f(x)', 'df(x)'])
    plt.show()


if __name__ == "__main__":
    # plotGraph()
    # simpleGradientDescent()
    # variousStartingLocationwithGradientDescent()
    varyingLearningRateWithGradientDescent()
    varyingLearningRatesAndTrainingEpocsWithGradientDescent()