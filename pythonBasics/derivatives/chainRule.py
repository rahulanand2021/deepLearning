import numpy as np
import sympy as sym
from rich import print
from sympy import pprint

def productRuleDerivative():
    x = sym.symbols('x')
    fx = 2*x**2
    gx = 4*x**3 - 3*x**4
    dfx = sym.diff(fx)
    dgx = sym.diff(gx)

    # Apply Product Rule
    manual = dfx*gx  + fx*dgx
    viasym = sym.diff(fx*gx)

    print("Manual :")
    pprint(manual)
    print("Via Sympy :")
    pprint(viasym)
    print((viasym))
    

if __name__ == "__main__":
    productRuleDerivative()