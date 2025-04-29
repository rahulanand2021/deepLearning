import numpy as np
import matplotlib.pyplot as plt


def createRandomSample():
    x = [1,2,4,6,5,4,0,-4,5,-2,6,10,-9,1,3,-6]
    populationMean = np.mean(x)
    sample = np.random.choice(x,size=5,replace=True)
    sampleMean = np.mean(sample)
    print(populationMean)
    print(sampleMean)

def createLargeRandomSample():
    x = [1,2,4,6,5,4,0,-4,5,-2,6,10,-9,1,3,-6]
    populationMean = np.mean(x)
    noOfExperiments = 10000
    sampleMeans = np.zeros(noOfExperiments)

    for index in range(noOfExperiments):
        sample = np.random.choice(x,size=5,replace=True)
        sampleMeans[index] = np.mean(sample)
        
    print(sampleMeans)  

    plt.hist(sampleMeans, bins=40, density=True)
    plt.plot([populationMean,populationMean],[0,.25], 'm--')
    plt.ylabel('Count')
    plt.xlabel('Sample Mean')
    plt.show()

if __name__ == "__main__":
    createRandomSample()
    createLargeRandomSample()