
import torch.nn.functional as F
import numpy as np
import torch

def calculateMinMaxVectorWithNumpy():
    v = np.array([1, 40, 2, -3])
    minVal = np.min(v)
    maxVal = np.max(v)

    print("Min Arg, Max Arg = %g , %g" %(minArg, maxArg))

def calculateMinMaxVectorWithPyTorch():
    v = torch.tensor([1, 40, 2, -3])
    minVal = torch.min(v)
    maxVal = torch.max(v)
    print(v)
    print("Min , Max = %g , %g" %(minVal, maxVal))
    minArg = torch.argmin(v)
    maxArg = torch.argmax(v)
    print("Min Arg, Max Arg = %g , %g" %(minArg, maxArg))
def calculateMinMaxMatrixWithNumpy() :
    M = np.array([   
                    [0,1,10],
                    [20,8,5]
                ])
    print(M) , print('  ')
    minVals1 = np.min(M)
    minVals2 = np.min(M, axis=0)  # Min in each column (Across Rows)
    minVals3 = np.min(M, axis=1)  # Min in each row (across Column)
    print(minVals1)
    print(minVals2)
    print(minVals3)
    minindex1 = np.argmin(M)
    minindex2 = np.argmin(M, axis=0)  # Min in each column (Across Rows)
    minindex3 = np.argmin(M, axis=1)  # Min in each row (across Column)
    print(minindex1)
    print(minindex2)
    print(minindex3)
def calculateMinMaxMatrixWithPyTorch() :
    M = torch.tensor([   
                    [0,1,10],
                    [20,8,5]
                ])
    print(M) , print('  ')
    minVals1 = torch.min(M)
    minVals2 = torch.min(M, axis=0)  # Min in each column (Across Rows)
    minVals3 = torch.min(M, axis=1)  # Min in each row (across Column)
    
    print(minVals1)
    print(minVals2.values)
    print(minVals3.values)
    
    
    minindex1 = torch.argmin(M)
    minindex2 = torch.argmin(M, axis=0)  # Min in each column (Across Rows)
    minindex3 = torch.argmin(M, axis=1)  # Min in each row (across Column)
    print(minindex1)
    print(minindex2)
    print(minindex3)
if __name__ == "__main__":
    # calculateMinMaxVectorWithNumpy()
    # calculateMinMaxMatrixWithNumpy()
    # calculateMinMaxVectorWithPyTorch()
    calculateMinMaxMatrixWithPyTorch()