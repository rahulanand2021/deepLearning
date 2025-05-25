import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
nPerCluster = 100
numberOfEpochs = 1000

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
    # fig = plt.figure(figsize=(5,5))
    # plt.plot(data[np.where(labels==0)[0],0],data[np.where(labels==0)[0],1],'bs')
    # plt.plot(data[np.where(labels==1)[0],0],data[np.where(labels==1)[0],1],'ko')
    # plt.title('The qwerties!')
    # plt.xlabel('qwerty dimension 1')
    # plt.ylabel('qwerty dimension 2')
    # plt.show()

    return data , labels

def buildArtificialNeuralNetwork(learningRate):

    ANNClassify = nn.Sequential(
                    nn.Linear(2,1),     # Input Layer
                    nn.ReLU(),            # Activation Function
                    nn.Linear(1,1)  
                )
    
    lossFunction = nn.BCEWithLogitsLoss()  # Loss Function
    optimizer = torch.optim.SGD(ANNClassify.parameters(),lr=learningRate)  # Flavour for Gradient Descent

    return ANNClassify, lossFunction, optimizer

    # numberOfEpochs = 1000
    # losses = np.zeros(numberOfEpochs)

    # for epoch_i in range(numberOfEpochs):
    #     yHat = ANNClassify(data)                 # Forward Pass

    #     # Compute Loss
    #     loss = lossFunction(yHat,labels)
    #     losses[epoch_i] = loss

    #     # BackProp
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    # # plt.plot(losses,'o',markerfacecolor='w',linewidth=.1)
    # # plt.xlabel('Epoch')
    # # plt.ylabel('Loss')
    # # plt.show()

    # predictions = ANNClassify(data)
    # predLabels  = predictions > 0.5
    # missClassified  = np.where(predLabels != labels)[0]
    # totalAccuracy = 100 - 100*len(missClassified)/(2*nPerCluster)
    # print('Final Accuracy %.2f%%' %totalAccuracy)

    # fig = plt.figure(figsize=(5,5))
    # plt.plot(data[missClassified,0] ,data[missClassified,1],'rx',markersize=12,markeredgewidth=3)
    # plt.plot(data[np.where(~predLabels)[0],0],data[np.where(~predLabels)[0],1],'bs')
    # plt.plot(data[np.where(predLabels)[0],0] ,data[np.where(predLabels)[0],1] ,'ko')

    # plt.legend(['Misclassified','blue','black'],bbox_to_anchor=(1,1))
    # plt.title(f'{totalAccuracy}% correct')
    # plt.show()
def trainTheModel(ANNModel, lossFunction, optimizer, data , labels):
    
    losses = torch.zeros(numberOfEpochs)

    for epoch_i in range(numberOfEpochs):
        yHat = ANNModel(data)                 # Forward Pass

        # Compute Loss
        loss = lossFunction(yHat,labels)
        losses[epoch_i] = loss

        # BackProp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    predictions = ANNModel(data)
    totalAccuracy = 100*torch.mean(((predictions>0) == labels).float())
    # print(totalAccuracy)
    return losses, predictions, totalAccuracy

def plotLosses(losses, totalAccuracy) :
    plt.plot(losses.detach(), 'o')
    plt.ylabel("Losses")
    plt.xlabel("Epochs")
    plt.title(f'Total Accuracy {totalAccuracy}')
    plt.show()

def runLearningRateExperiment(data, lables):
    
    learningRates = np.linspace(0.01,.1, 40)
    accByLearningRate = []
    allLosses = torch.zeros((len(learningRates), numberOfEpochs))

    for index, lr in enumerate(learningRates):
        ANNModel, lossFunction, optimizer = buildArtificialNeuralNetwork(learningRate=lr)
        losses, predictions, totalAccuracy = trainTheModel(ANNModel,lossFunction,optimizer, data, lables)
        accByLearningRate.append(totalAccuracy)
        allLosses[index,:] = losses.detach()

    print(sum(torch.tensor(accByLearningRate)>70)/len(accByLearningRate))

    fig, ax = plt.subplots(1,2,figsize=(12,4))
    ax[0].plot(learningRates, accByLearningRate, 's-')
    ax[0].set_xlabel('Learning Rate')
    ax[0].set_ylabel('Accuracy Rate')
    ax[0].set_title('Accuracy by Learning Rate')

    ax[1].plot(allLosses.T)
    ax[1].set_xlabel('Epochs Number')
    ax[1].set_ylabel('Loss')
    ax[1].set_title('Losses by Learning Rate')    
    plt.show()

def runMetaLearningRateExperiment(data, lables):
    
    numberOfExpMeta = 50

    learningRates = np.linspace(0.01,.1, 40)
    accuracyMeta = np.zeros((numberOfExpMeta, len(learningRates)))

    for experi in range (numberOfExpMeta) :
        print(f'Running Meta Experiment for Learing Rate for {experi} time')
        for index, lr in enumerate(learningRates):
            ANNModel, lossFunction, optimizer = buildArtificialNeuralNetwork(learningRate=lr)
            losses, predictions, totalAccuracy = trainTheModel(ANNModel,lossFunction,optimizer, data, lables)
            accuracyMeta[experi, index] = totalAccuracy

    plt.plot(learningRates, np.mean(accuracyMeta, axis=0), 's-')
    plt.xlabel('Learning Rates')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Learning Rate')
    plt.show()

if __name__ == "__main__":
    learningRate = 0.1
    data , labels = buildCategoricalData()
    # ANNModel, lossFunction, optimizer = buildArtificialNeuralNetwork(learningRate=learningRate)
    # losses, predictions, totalAccuracy = trainTheModel(ANNModel, lossFunction, optimizer,data, labels)
    # plotLosses(losses, totalAccuracy)
    # runLearningRateExperiment(data=data, lables=labels)
    runMetaLearningRateExperiment(data=data, lables=labels)