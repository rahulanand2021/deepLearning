import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import sympy.plotting.plot as symplot
from sympy.plotting.plot import MatplotlibBackend

def simpleDerivative():

    x = sym.symbols('x')
    fx = 2*x**2
    df = sym.diff(fx,x)

    p = symplot(fx, (x, -5, 5), label="The Function", show=False, line_color='blue')
    p.extend(symplot(df, (x, -5, 5), label="Differentiation", show=False, line_color='red'))
    p.legend = True
    p.title = "The Functions with its Differentiations"
    p.show()

def reluAndSigmoidDerivatives():
    x = sym.symbols('x')
    relu = sym.Max(0,x)
    sigmoid = 1 / (1 + sym.exp(-x))
    
    p = symplot(relu, (x, -5, 5), label="ReLU", show=False, line_color='blue')
    p.extend(symplot(sigmoid, (x, -5, 5), label="Sigmoid", show=False, line_color='red'))
    p.legend = True
    p.title = "The Functions"
    p.show()

if __name__ == "__main__":
    simpleDerivative()
    # reluAndSigmoidDerivatives()