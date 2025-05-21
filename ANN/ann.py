import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def createDataAndShowPlot():
    N = 50
    x = torch.randn(N,1)
    y = x + torch.randn(N,1)/2
    # plt.plot(x,y,'ro')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()
    return x, y

def createDataCodeChallengeAndShowPlot(m):
    N = 50
    x = torch.randn(N,1)
    y = m*x + torch.randn(N,1)/2
    # plt.plot(x,y,'ro')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()
    return x, y

def buildArtificialNeuralNetwork(x, y):

    ANNreg = nn.Sequential(
                    nn.Linear(1,1),     # Input Layer
                    nn.ReLU(),            # Activation Function
                    nn.Linear(1,1)      # Output
                )
    # print(ANNreg)
    learningRate = 0.05
    lossFunction = nn.MSELoss()   # Loss Function
    optimizer = torch.optim.SGD(ANNreg.parameters(),lr=learningRate)  # Flavour for Gradient Descent

    numberOfEpochs = 50
    losses = np.zeros(numberOfEpochs)

    for epoch_i in range(numberOfEpochs):
        yHat = ANNreg(x)                 # Forward Pass

        # Compute Loss
        loss = lossFunction(yHat,y)
        losses[epoch_i] = loss

        # BackProp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    predictions = ANNreg(x)

    # print('******** LOSSES ***************')
    # print(losses)

    # print('**********PREDICTIONS**********')
    # print(predictions)

    testLoss = (predictions - y).pow(2).mean()
    print(predictions.detach())

    # plt.plot(losses,'o',markerfacecolor='w',linewidth=.1)
    # plt.plot(numberOfEpochs,testLoss.detach(),'ro')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Final loss = %g' %testLoss.item())
    # plt.show()

    # plt.plot(x,y, 'bo', label="Real Data")
    # plt.plot(x, predictions.detach(), 'rs', label="Predictions")
    # plt.title(f'Predicting Data r ={np.corrcoef(y.T,predictions.detach().T)[0,1]}')
    # plt.legend()
    # plt.show()
    return losses, predictions
    
if __name__ == "__main__":
    x, y =  createDataAndShowPlot()
    buildArtificialNeuralNetwork(x , y)
    # losses_final = np.zeros((21,50))
    # predictions_final = np.zeros((21,50))
    # m = np.linspace(-2,2,21)
    # for idx, slope in enumerate(m):
    #     print(slope)
    #     x, y = createDataCodeChallengeAndShowPlot(slope)
    #     loss, predict = buildArtificialNeuralNetwork(x , y)
    # print("Final Loss List of List")
    # print(losses_final)


