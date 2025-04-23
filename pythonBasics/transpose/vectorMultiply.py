import numpy as np
import torch

def vecMulWithNumPy():
    np1 = np.array( [1,2,3,4] )
    np2 = np.array( [0,2,0, 1] )
    print(np.dot(np1,np2))
    print(np.sum(np1*np2))

def vecMulWithPyTorch():
    tv1 = torch.tensor( [1,2,3,4] )
    tv2 = torch.tensor( [0,2,0, 1] )
    print(torch.dot(tv1,tv2))
    print(torch.sum(tv1*tv2))

if __name__ == "__main__" :
    vecMulWithNumPy()
    vecMulWithPyTorch()