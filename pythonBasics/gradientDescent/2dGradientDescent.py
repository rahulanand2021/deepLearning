import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from sympy import pprint

def peaks(x,y):
    x,y = np.meshgrid(x,y)

    z = 3*(1-x)**2 * np.exp(-(x**2) - (y+1)**2) \
        - 10*(x/5 - x**3 - y**5) * np.exp(-x**2-y**2) \
        - 1/3*np.exp(-(x+1)**2 - y**2)
    return z

def plottingTwoDimensionFunction():
    x = np.linspace(-3,3,201)
    y = np.linspace(-3,3,201)

    Z = peaks(x,y)
    plt.imshow(Z, extent=[x[0], x[-1],y[0], y[-1]], vmin=-5, vmax=5, origin='lower')
    plt.show()

def derivativesUsingSympy():
    sx, sy = sym.symbols('sx','sy')
    
    sZ = 3*(1-sx)**2 * sym.exp(-(sx**2) - (sy+1)**2) \
        - 10*(sx/5 - sx**3 - sy**5) * sym.exp(-sx**2-sy**2) \
        - 1/3*sym.exp(-(sx+1)**2 - sy**2)
    
    partialDifferentiation_x = sym.lambdify((sx,sy), sym.diff(sZ,sx))
    partialDifferentiation_y = sym.lambdify((sx,sy), sym.diff(sZ,sy))

if __name__ == "__main__":
    plottingTwoDimensionFunction()
    #  twoDimensionGradientDescent()