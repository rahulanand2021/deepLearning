import numpy as np
import matplotlib as plt
import torch
import torch.nn.functional as F


def calculateMinMaxOfVectorWithNumpy():
    v = np.array([1, 40, 2, -3])
    minVal = np.min(v)
    maxVal = np.max(v)
    print(v)
    print("Min , Max = %g , %g" %(minVal, maxVal))
    minArg = np.argmin(v)
    maxArg = np.argmax(v)
    print("Min Arg, Max Arg = %g , %g" %(minArg, maxArg))

def calculateMinMaxOfMatrixWithNumpy():
    M = np.array([   [0,1,10],
                     [20,8,5]
                  ])
    minVals1 = np.min(M)
    minVals2 = np.min(M, axis=0)   # Min for each column  (across row)
    minVals3 = np.min(M, axis=1)   # Min for each Row (across column)

    print(minVals1)
    print(minVals2)
    print(minVals3)

if __name__ == "__main__":
    calculateMinMaxOfVectorWithNumpy()
    calculateMinMaxOfMatrixWithNumpy()