import numpy as np
import gzip
from sys import path
path.insert(1, "..")
from sobek.network import network

print("--- Data loading ---")

def getData(fileName):
    with open(fileName, 'rb') as f:
        data = f.read()
        return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()


tempTrainImages = getData("./MNIST/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 784)).tolist()
trainImages = []
for image in tempTrainImages:
    for pixel in range(784):
        if image[pixel] !=0:
            image[pixel] = image[pixel]/256
    trainImages.append(np.array(image, dtype=np.float64))
tempTrainLabels = getData("./MNIST/t10k-labels-idx1-ubyte.gz")[8:]
trainLabels = []
for label in tempTrainLabels:
    trainLabels.append(np.zeros(10))
    trainLabels[-1][label] = 1.0

print("--- Testing ---")

myNetwork = network.networkFromFile("MNIST30epoch")

print(myNetwork.accuracy(trainImages, trainLabels))