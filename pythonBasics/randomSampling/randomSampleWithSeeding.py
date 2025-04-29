import numpy as np
import torch

def createRandomSampleWithSeed():
    np.random.seed(29)
    print(np.random.randn(5))
    print(np.random.randn(5))

def createRandomSampleWithRandomState():
    randSeed1 = np.random.RandomState(29)
    randSeed2 = np.random.RandomState(20250429)

    print(randSeed1.randn(5))
    print(randSeed2.randn(5))
    print(randSeed1.randn(5))
    print(randSeed2.randn(5))
    print(np.random.randn(5))


def createRandomSampleWithTorch():
    print(torch.randn(5))
    torch.manual_seed(29)
    print(torch.randn(5))

if __name__ == "__main__":
    # createRandomSampleWithSeed()
    # createRandomSampleWithRandomState()   # More Prefered than above
    createRandomSampleWithTorch()