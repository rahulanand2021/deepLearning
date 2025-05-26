import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
nPerCluster = 100

def buildCategoricalData():
    
    blur = 1

    A = [  1, 1]
    B = [  5, 1]
    C = [  3, -2 ]

    # generate data
    # generate data
    a = [ A[0]+np.random.randn(nPerCluster)*blur , A[1]+np.random.randn(nPerCluster)*blur ]
    b = [ B[0]+np.random.randn(nPerCluster)*blur , B[1]+np.random.randn(nPerCluster)*blur ]
    c = [ C[0]+np.random.randn(nPerCluster)*blur , C[1]+np.random.randn(nPerCluster)*blur ]

    # true labels   
    labels_np = np.vstack((np.zeros((nPerCluster,1)),np.ones((nPerCluster,1)),1+np.ones((nPerCluster,1))))

    # concatanate into a matrix
    data_np = np.hstack((a,b,c)).T

    # convert to a pytorch tensor
    data = torch.tensor(data_np).float()
    labels = torch.squeeze(torch.tensor(labels_np).long())

        # show the data
    # fig = plt.figure(figsize=(5,5))
    # plt.plot(data[np.where(labels==0)[0],0],data[np.where(labels==0)[0],1],'bs')
    # plt.plot(data[np.where(labels==1)[0],0],data[np.where(labels==1)[0],1],'ko')
    # plt.plot(data[np.where(labels==2)[0],0],data[np.where(labels==2)[0],1],'ro')
    # plt.title('The qwerties!')
    # plt.xlabel('qwerty dimension 1')
    # plt.ylabel('qwerty dimension 2')
    # # plt.show()

    print(labels)
    return data , labels

def createANNModel():
    ANNMultiClass = nn.Sequential(
                nn.Linear(2,64),
                nn.ReLU(),
                nn.Linear(64,64),
                nn.ReLU(),
                nn.Linear(64,3)
    )

    lossFunction = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ANNMultiClass.parameters(),lr=.1)  # Flavour for Gradient Descent

    return ANNMultiClass, lossFunction, optimizer

def trainTheModel(ANNMultiClass, lossFunction, optimizer, data, labels):
    
    numberOfEpochs = 3000

    losses = torch.zeros(numberOfEpochs)
    ongoingAccuracy = []
    yHat  = None
    for epoch_i in range(numberOfEpochs):
        yHat = ANNMultiClass(data)                 # Forward Pass

        # Compute Loss
        loss = lossFunction(yHat,labels)
        losses[epoch_i] = loss

        # BackProp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute Accuracy
        matches = torch.argmax(yHat, axis=1) == labels
        matchNumeric = matches.float()
        accuracyPct = 100*torch.mean(matchNumeric)
        ongoingAccuracy.append(accuracyPct)

    predictions = ANNMultiClass(data)
    predLabel = torch.argmax(predictions, axis = 1)
    totalAccuracy = 100*torch.mean((predLabel == labels).float())
    print(f'Total Accuracys is {totalAccuracy} %')
    sm = nn.Softmax(1)
    # print(yHat)                         # Show the Raw Predictions
    # print(torch.sum((yHat), axis=1))    # Show the output of 150 data points, summed across rows, axis = 1
    # print(torch.sum(sm(yHat), axis=1))  # Just to check that the Softmax of the 3 output is equal to 1


    fig, ax = plt.subplots(1,2,figsize=(13,4))
    ax[0].plot(losses.detach())
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Losses')

    ax[1].plot(ongoingAccuracy)
    ax[1].set_xlabel('Epochs Number')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Accuracy')    
    plt.show()

    # Plotting the Raw Data 
    # fig = plt.figure(figsize=(13,4))
    # plt.plot(sm(yHat.detach()), 's-', markerfacecolor='w')
    # plt.xlabel('Stimulus Number')
    # plt.ylabel('Probability')
    # plt.legend(['0', '1', '2'])
    # plt.show()

if __name__ == "__main__":
    data , labels = buildCategoricalData()
    ANNMultiClass, lossFunction, optimizer = createANNModel()
    trainTheModel(ANNMultiClass, lossFunction, optimizer, data, labels)