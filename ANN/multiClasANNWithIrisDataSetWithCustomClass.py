import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from  customANNClassforIrisDataSet import customANNClassForIris

def loadDataSet():
    iris = sns.load_dataset('iris')
    return iris

def transformPandasToTensor(iris_data):
    data = torch.tensor(iris_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values, dtype=torch.float32)
    # print(features.shape)
    labels = torch.zeros(len(data), dtype=torch.long)
    labels[iris_data.species=='setosa'] = 0
    labels[iris_data.species=='versicolor'] = 1
    labels[iris_data.species=='virginica'] = 2

    return data , labels

def createANNModel():
    numberOfUnitsPerLayer = 12
    numberOfLayers = 4

    ANNiris = customANNClassForIris(numberOfUnitsPerLayer, numberOfLayers)
    lossFunction = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ANNiris.parameters(),lr=.01)  # Flavour for Gradient Descent

    return ANNiris, lossFunction, optimizer

def testTheModel(ANNiris):
    tmpx = torch.randn(10,4)
    y = ANNiris(tmpx)
    print(y.shape)
    print(y)

if __name__ == "__main__":
    # iris_data = loadDataSet()
    createANNModel()
    # data , labels = transformPandasToTensor(iris_data)
    ANNiris, lossFunction, optimizer = createANNModel()
    testTheModel(ANNiris)
    # trainTheModel(ANNiris, lossFunction, optimizer, data, labels)