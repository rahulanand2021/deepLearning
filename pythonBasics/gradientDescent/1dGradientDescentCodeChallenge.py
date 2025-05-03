import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import sympy.plotting.plot as symplot
from sympy import pprint

x = sym.symbols('x')
fx_expr = sym.cos(2 * sym.pi * x) + x**2
dfx_expr = sym.diff(fx_expr, x)

fx_numeric = sym.lambdify(x, fx_expr, 'numpy')
dfx_numeric = sym.lambdify(x, dfx_expr, 'numpy')

def simpleDerivative():
    x = sym.symbols('x')
    fx = sym.cos(2*sym.pi*x) + x**2
    df = sym.diff(fx,x)
    pprint(fx)
    pprint(df)
    p = symplot(fx, (x, -2, 2), label="The Function", show=False, line_color='blue')
    p.extend(symplot(df, (x, -2, 2), label="Differentiation", show=False, line_color='red'))
    p.legend = True
    p.title = "The Functions with its Differentiations"
    p.show()


# def fx(x):
#     # x = sym.symbols('x')
#     result = (sym.cos(2*sym.pi*x) + x**2)
#     return result

def derivativeOf(x1):
    x = sym.symbols('x')
    fx = sym.cos(2*sym.pi*x) + x**2
    df = sym.diff(fx,x)
    result = df.subs(x, x1) 
    return result

def plottingFunctionAndDerivatives():
    x = np.linspace(-2,2,2001)
    plt.plot(x,fx_numeric(x), x, dfx_numeric(x))
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
    training_epocs = 1000
    print(f"Initial Local Minimum (Guess) : {localMinimum}")
    for index in range(training_epocs):
        grad = dfx_numeric(localMinimum)
        localMinimum = localMinimum - grad*learning_rate
    print(f"After GD, Actual LocalMinimum : {localMinimum}")

def oneDimensionGradientDescentToPlotGraph():
    x = np.linspace(-3,3,2001)
    # localMinimum = np.random.choice(x,1)
    localMinimum = np.random.choice(x,1)
    learning_rate = 0.03
    training_epocs = 100
    print(f"Initial Local Minimum (Guess) : {localMinimum}")
    
    modelparams = np.zeros((training_epocs,2))
    # for index in range(training_epocs):
    #     grad = derivativeOf(localMinimum)
    #     localMinimum = localMinimum - grad*learning_rate
    #     modelparams[index,0] = localMinimum[0]
    #     modelparams[index,1] = grad[0]
    # print(f"After GD, Actual LocalMinimum : {localMinimum}")

    for index in range(training_epocs):
        grad = dfx_numeric(localMinimum)
        localMinimum = localMinimum - grad*learning_rate
        modelparams[index,0] = localMinimum[0]
        modelparams[index,1] = grad[0]
    print(f"After GD, Actual LocalMinimum : {localMinimum}")

    # plot the gradient over iterations

    fig,ax = plt.subplots(2,1,figsize=(10,10))

    for i in range(2):
        ax[i].plot(modelparams[:,i],'o-')
        ax[i].set_xlabel('Iteration')
        ax[i].set_title(f'Final estimated minimum: {localMinimum[0]:.5f}')

    ax[0].set_ylabel('Local minimum')
    ax[1].set_ylabel('Derivative')

    plt.show()
if __name__ == "__main__":
    #  plottingFunctionAndDerivatives()
    # oneDimensionGradientDescent()
    oneDimensionGradientDescentToPlotGraph()
    # simpleDerivative()
    # print(fx(100))
