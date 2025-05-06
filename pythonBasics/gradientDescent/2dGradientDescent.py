import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from sympy import pprint

partialDifferentiation_x = None
partialDifferentiation_y = None

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
    global partialDifferentiation_x
    global partialDifferentiation_y

    sx,sy = sym.symbols('sx,sy')
    
    sZ = 3*(1-sx)**2 * sym.exp(-(sx**2) - (sy+1)**2) \
        - 10*(sx/5 - sx**3 - sy**5) * sym.exp(-sx**2-sy**2) \
        - 1/3*sym.exp(-(sx+1)**2 - sy**2)
    
    partialDifferentiation_x = sym.lambdify((sx,sy), sym.diff(sZ,sx),'sympy')
    partialDifferentiation_y = sym.lambdify((sx,sy), sym.diff(sZ,sy),'sympy')

    #Test the function above
    # print(partialDifferentiation_x(1,1).evalf())
    # print(partialDifferentiation_y(1,1).evalf())

def twoDimensionGradientDescent():
    localminimum = (np.random.rand(2) * 4 - 2)    # Force the value to be between -2 and +2
    startPoint = localminimum[:]   # Make a copy for later

    learning_rate = 0.01
    training_epoch = 1000

    trajectory = np.zeros((training_epoch,2))
    for index in range(training_epoch):
        grad = np.array([
                        partialDifferentiation_x(localminimum[0],localminimum[1]),
                        partialDifferentiation_y(localminimum[0],localminimum[1])
                        ])
        localminimum = localminimum - grad*learning_rate
        trajectory[index,:] = localminimum
    
    print(trajectory)
    print(localminimum)
    print(startPoint)

    # Visualize the above
    x = np.linspace(-3,3,201)
    y = np.linspace(-3,3,201)
    Z = peaks(x,y)
    plt.imshow(Z, extent=[x[0], x[-1],y[0], y[-1]], vmin=-5, vmax=5, origin='lower')
    plt.plot(startPoint[0],startPoint[1],'bs')
    plt.plot(localminimum[0],localminimum[1],'ro')
    plt.plot(trajectory[: , 0],trajectory[: , 1],'r')
    plt.legend(['Random Start', 'Local Minimum'])
    plt.colorbar()
    plt.show()
 
if __name__ == "__main__":
    # plottingTwoDimensionFunction()
    derivativesUsingSympy()
    twoDimensionGradientDescent()