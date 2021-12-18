import numpy as np
import random
from sobek.network import network

random.seed()

myNetwork = network(10, 10)

learningRate = 3

for j in range(10000):
    rand = []
    inputs = []
    desiredOutputs = []
    
    if (j%50 == 0):
        print(j)

    for i in range(10):
        rand.append( random.randrange(10)/10)

    for i in range(10):
        desiredOutputs.append(np.zeros(10))
        desiredOutputs[i][9 - int(rand[i]*10)] = 1.0

    for i in range(10):
        inputs.append(np.zeros(10))
        inputs[i][int(rand[i]*10)] = 1.0
    
    myNetwork.train(inputs, desiredOutputs, learningRate)

test = []
test.append(np.zeros(10))
test.append(np.zeros(10))
test[0][1] = 1.0
test[1][5] = 1.0
print(test[0])
print(myNetwork.process(test[0]))
print(myNetwork.process(test[1]))