import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns


def loadDataSet():
    iris = sns.load_dataset('iris')
    return iris

def showPairPlot(iris_data):
    sns.pairplot(iris_data, hue='species')
    plt.show()

def transformPandasToTensor(iris_data):
    data = torch.tensor(iris_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values, dtype=torch.float32)
    # print(features.shape)
    labels = torch.zeros(len(data), dtype=torch.long)
    labels[iris_data.species=='setosa'] = 0
    labels[iris_data.species=='versicolor'] = 1
    labels[iris_data.species=='virginica'] = 2

    return data , labels

def createANNModel():
    ANNiris = nn.Sequential(
                nn.Linear(4,64),
                nn.ReLU(),
                nn.Linear(64,64),
                nn.ReLU(),
                nn.Linear(64,3)
    )

    lossFunction = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ANNiris.parameters(),lr=.01)  # Flavour for Gradient Descent

    return ANNiris, lossFunction, optimizer

def createANNModel(hiddenLayer):
    ANNiris = nn.Sequential(
                nn.Linear(4,hiddenLayer),
                nn.ReLU(),
                nn.Linear(hiddenLayer,hiddenLayer),
                nn.ReLU(),
                nn.Linear(hiddenLayer,3)
    )

    lossFunction = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ANNiris.parameters(),lr=.01)  # Flavour for Gradient Descent

    return ANNiris, lossFunction, optimizer

def trainTheModel(ANNModeliris, lossFunction, optimizer, data, labels):
    
    numberOfEpochs = 1000

    losses = torch.zeros(numberOfEpochs)
    ongoingAccuracy = []
    yHat  = None
    for epoch_i in range(numberOfEpochs):
        yHat = ANNModeliris(data)                 # Forward Pass

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

    predictions = ANNModeliris(data)
    predLabel = torch.argmax(predictions, axis = 1)
    totalAccuracy = 100*torch.mean((predLabel == labels).float())
    print(f'Total Accuracys is {totalAccuracy} % ')
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
    # plt.legend(['setosa', 'versicolor', 'virginica'])
    # plt.show()

def trainTheModelForDynamicHiddenLayer(ANNModeliris, lossFunction, optimizer, data, labels, hiddenLayer):
    
    numberOfEpochs = 1000

    losses = torch.zeros(numberOfEpochs)
    yHat  = None

    for epoch_i in range(numberOfEpochs):
        yHat = ANNModeliris(data)                 # Forward Pass

        # Compute Loss
        loss = lossFunction(yHat,labels)
        losses[epoch_i] = loss

        # BackProp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    predictions = ANNModeliris(data)
    predLabel = torch.argmax(predictions, axis = 1)
    totalAccuracy = 100*torch.mean((predLabel == labels).float())
    print(f'Total Accuracys is {totalAccuracy} % for hidden Layer {hiddenLayer}')
    return totalAccuracy

def trainModelOverDynamicHiddenLayer(data, labels):
    #noOfExperiments = 150
    noOfHiddenLayer = np.arange(1,129)
    accuraciesOfGivenLayer = []

    for hiddenLayer in noOfHiddenLayer:
        ANNModeliris, lossFunction, optimizer = createANNModel(hiddenLayer)
        acc = trainTheModelForDynamicHiddenLayer(ANNModeliris, lossFunction, optimizer, data, labels, hiddenLayer)
        accuraciesOfGivenLayer.append(acc)
    
    fig, ax = plt.subplots(1, figsize=(12,6))
    ax.plot(accuraciesOfGivenLayer, 'ko-', markerfacecolor='w', markersize=9)
    ax.set_xlabel('Number of Hidden Unit')
    ax.set_ylabel('Accuracies')
    ax.set_title("Accuracy Chart!")
    plt.show()

if __name__ == "__main__":
    iris_data = loadDataSet()
    # showPairPlot(iris_data)
    data , labels = transformPandasToTensor(iris_data)
    # ANNiris, lossFunction, optimizer = createANNModel()
    # trainTheModel(ANNiris, lossFunction, optimizer, data, labels)
    trainModelOverDynamicHiddenLayer(data,labels)