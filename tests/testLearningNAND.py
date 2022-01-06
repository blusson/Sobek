import numpy as np
import random
import time
from sys import path
path.insert(1, "..")
from sobek.network import network

random.seed()

myNetwork = network(2, 2, 1)

learningRate = 3

test = []
result = []
test.append(np.zeros(2))
test.append(np.zeros(2))
test.append(np.zeros(2))
test.append(np.zeros(2))
test[1][1] = 1.0
test[2][0] = 1.0
test[3][0] = 1.0
test[3][1] = 1.0
result.append(np.ones(1))
result.append(np.ones(1))
result.append(np.ones(1))
result.append(np.zeros(1))

learningTime = 0

nbRep = 1

for i in range(nbRep):
    if (i%(nbRep/10) == 0): print(i)

    startTime = time.perf_counter()

    myNetwork.train(test, result, learningRate, len(test), 10000, visualize=True)

    endTime = time.perf_counter()
    learningTime += endTime - startTime
learningTime = learningTime / nbRep

test = []
result = []
test.append(np.zeros(2))
test.append(np.zeros(2))
test.append(np.zeros(2))
test.append(np.zeros(2))
test[1][1] = 1.0
test[2][0] = 1.0
test[3][0] = 1.0
test[3][1] = 1.0
result.append(np.ones(1))
result.append(np.ones(1))
result.append(np.ones(1))
result.append(np.zeros(1))

#print(myNetwork.weights)
#print(myNetwork.biases)
print("0 0 : " + str(myNetwork.process(test[0])) + " == 1 ?")
print("0 1 : " + str(myNetwork.process(test[1])) + " == 1 ?")
print("1 0 : " + str(myNetwork.process(test[2])) + " == 1 ?")
print("1 1 : " + str(myNetwork.process(test[3])) + " == 0 ?")

myNetwork.saveToFile("NAND")

print("Learning time : " + str(endTime - startTime))