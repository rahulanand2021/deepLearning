import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
nPerCluster = 100

def buildCategoricalData():
    
    blur = 1

    A = [  1, 1 ]
    B = [  5, 1 ]

    # generate data
    a = [ A[0]+np.random.randn(nPerCluster)*blur , A[1]+np.random.randn(nPerCluster)*blur ]
    b = [ B[0]+np.random.randn(nPerCluster)*blur , B[1]+np.random.randn(nPerCluster)*blur ]

    # true labels
    labels_np = np.vstack((np.zeros((nPerCluster,1)),np.ones((nPerCluster,1))))

    # concatanate into a matrix
    data_np = np.hstack((a,b)).T

    # convert to a pytorch tensor
    data = torch.tensor(data_np).float()
    labels = torch.tensor(labels_np).float()

    # show the data
    fig = plt.figure(figsize=(5,5))
    plt.plot(data[np.where(labels==0)[0],0],data[np.where(labels==0)[0],1],'bs')
    plt.plot(data[np.where(labels==1)[0],0],data[np.where(labels==1)[0],1],'ko')
    plt.title('The qwerties!')
    plt.xlabel('qwerty dimension 1')
    plt.ylabel('qwerty dimension 2')
    plt.show()

    return data , labels

def buildArtificialNeuralNetwork(data, labels):

    ANNClassify = nn.Sequential(
                    nn.Linear(2,1),     # Input Layer
                    nn.ReLU(),            # Activation Function
                    nn.Linear(1,1),     
                    nn.Sigmoid()          # Output
                )
    
    learningRate = 0.1
    lossFunction = nn.BCELoss()  # Loss Function
    optimizer = torch.optim.SGD(ANNClassify.parameters(),lr=learningRate)  # Flavour for Gradient Descent

    numberOfEpochs = 1000
    losses = np.zeros(numberOfEpochs)

    for epoch_i in range(numberOfEpochs):
        yHat = ANNClassify(data)                 # Forward Pass

        # Compute Loss
        loss = lossFunction(yHat,labels)
        losses[epoch_i] = loss

        # BackProp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # plt.plot(losses,'o',markerfacecolor='w',linewidth=.1)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.show()

    predictions = ANNClassify(data)
    predLabels  = predictions > 0.5
    missClassified  = np.where(predLabels != labels)[0]
    totalAccuracy = 100 - 100*len(missClassified)/(2*nPerCluster)
    print('Final Accuracy %.2f%%' %totalAccuracy)

    fig = plt.figure(figsize=(5,5))
    plt.plot(data[missClassified,0] ,data[missClassified,1],'rx',markersize=12,markeredgewidth=3)
    plt.plot(data[np.where(~predLabels)[0],0],data[np.where(~predLabels)[0],1],'bs')
    plt.plot(data[np.where(predLabels)[0],0] ,data[np.where(predLabels)[0],1] ,'ko')

    plt.legend(['Misclassified','blue','black'],bbox_to_anchor=(1,1))
    plt.title(f'{totalAccuracy}% correct')
    plt.show()

def computePredictions(data, labels, annClassify):
    predictions = annClassify(data)
    print(type(predictions))
    # print(predictions)

if __name__ == "__main__":
    data , labels = buildCategoricalData()
    buildArtificialNeuralNetwork(data=data, labels=labels)
    # computePredictions(data, labels, annClassify)