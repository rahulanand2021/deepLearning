import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def createDataAndShowPlot():
    N = 50
    x = torch.randn(N,1)
    y = x + torch.randn(N,1)/2
    plt.plot(x,y,'s')

    plt.show()

if __name__ == "__main__":
    createDataAndShowPlot()

