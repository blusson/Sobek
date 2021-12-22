import numpy as np
import gzip
import time
from sys import path
path.insert(1, "..")
from sobek.network import network


print("--- Data loading ---")

def getData(fileName):
    with open(fileName, 'rb') as f:
        data = f.read()
        return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()


tempTrainImages = getData("./MNIST/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 784)).tolist()
trainImages = []
for image in tempTrainImages:
    for pixel in range(784):
        if image[pixel] !=0:
            image[pixel] = image[pixel]/256
    trainImages.append(np.array(image, dtype=np.float64))
tempTrainLabels = getData("./MNIST/train-labels-idx1-ubyte.gz")[8:]
trainLabels = []
for label in tempTrainLabels:
    trainLabels.append(np.zeros(10))
    trainLabels[-1][label] = 1.0

myNetwork = network(784, 30, 10)

learningRate = 3.0

print("--- Learning ---")

startTime = time.perf_counter()

"""
for i in range(1):
    print("Epoch: " + str(i))
    batchEnd = 10
    while batchEnd < 1000:
        batchImages = trainImages[:batchEnd]
        batchLabels = trainLabels[:batchEnd]
        myNetwork.train(batchImages, batchLabels, learningRate)
        batchEnd += 10
        if (batchEnd%100) == 0:
            print(batchEnd)
"""

myNetwork.train(trainImages, trainLabels, learningRate, 10, 30)

endTime = time.perf_counter()

print("Learning time : " + str(endTime - startTime))

print(trainLabels[121])
print(myNetwork.process(trainImages[121]))

myNetwork.saveToFile("MNIST30epoch")