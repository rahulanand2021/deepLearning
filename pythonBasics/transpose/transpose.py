import numpy as np
import torch

def transposeVector():
    nv = np.array([ [1,2,3,4]])
    print(nv) , print(' ')
    print(nv.T), print(' ')
    nvT = nv.T
    print(nvT.T), print(' ')
    print(f"Variable nv is of type {type(nv)}")

def transposeMatrix():
    nm = np.array([ [1,2,3,4],
                    [5,6,7,8]])
    print(nm) , print(' ')
    print(nm.T), print(' ')
    nmT = nm.T
    print(nmT.T), print(' ')
    print(f"Variable nv is of type {type(nm)}")

def transposeVectorWithPyTorch():
    tv = torch.tensor([ [1,2,3,4]])
    print(tv) , print(' ')
    print(tv.T), print(' ')
    nvT = tv.T
    print(nvT.T), print(' ')
    print(f"Variable tv is of type {type(tv)}")

def transposeMatrixWithPyTorch():
    tm = torch.tensor([ [1,2,3,4],
                       [5,6,7,8]
                       ])
    print(tm) , print(' ')
    print(tm.T), print(' ')
    tmT = tm.T
    print(tmT.T), print(' ')
    print(f"Variable tm is of type {type(tm)}")

if __name__ == "__main__" :
    transposeVector()
    transposeMatrix()
    transposeVectorWithPyTorch()
    transposeMatrixWithPyTorch()