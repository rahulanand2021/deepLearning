import numpy as np
import matplotlib as plt
import torch
import torch.nn.functional as F

# H(p) = Σᵢ₌₁ⁿ p(xᵢ)*log(p(xᵢ))- Entropy

def calculateBinaryEntropyWithNumpy():

    # probability of single event happening is 0.25, hence that event not happening is 1 - 0.25
    x = [0.25, 0.75]
    H = 0
    for p in x:    
        H += -(p* np.log(p))
    print(str(H))

    # Explicitly written out for Entropy Function
    H_1 = -(p * np.log(p) + (1-p) * np.log(1-p))
    print(H_1)

def calculateBinaryCrossEntropyWithNumpy():
    p = [1 , 0]           # Probability that the picture is a Dog
    q = [0.25, 0.75]      # Probability of the Model predecting if its a Dog

    H = 0 

    for i in range (len(p)):
        H+= -p[i]*np.log(q[i])

    print('Cross Entropy ' + str(H))

    # Explicitly written out for Binary Cross Entropy Function
    H = 0 
    H = - (p[0]*np.log(q[0]) + p[1]*np.log(q[1]))

    print('Cross Entropy ' + str(H))

def calculateBinaryCrossEntropyWithPyTorch():
    p = [1 , 0]           # Probability that the picture is a Dog
    q = [0.25, 0.75]      # Probability of the Model predecting if its a Dog

    output = F.binary_cross_entropy(torch.Tensor(q), torch.Tensor(p))   # Note that q comes first and then p
    print(output)

if __name__ == "__main__":
    # calculateBinaryEntropyWithNumpy()
    calculateBinaryCrossEntropyWithNumpy()
    calculateBinaryCrossEntropyWithPyTorch()