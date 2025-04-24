import numpy as np
import matplotlib as plt
import torch
import torch.nn.functional as F


def calculateMinMaxWithNumpy():
    v = np.array([1, 40, 2, -3])
    minVal = np.min(v)
    maxVal = np.max(v)

    print(v)
    print("Min , Max = %g , %g" %(minVal, maxVal))

    minArg = np.argmin(v)
    maxArg = np.argmax(v)

    print("Min Arg, Max Arg = %g , %g" %(minArg, maxArg))

if __name__ == "__main__":
    calculateMinMaxWithNumpy()