#!/bin/python3
from sys import path
path.insert(1, "..")
from sobek.network import network
import pickle

with open("flowerGardenData", "rb") as file:
    data = pickle.load(file)

trainPoints = data[0]
trainLabels = data[1]

myNetwork = network(2, 16, 1)

learningRate = 5.0

myNetwork.train(trainPoints, trainLabels, learningRate, batchSize=10, epochs=1000, visualize=True)
