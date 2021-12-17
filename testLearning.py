import numpy as np
import random
from sobek.network import network

random.seed()

myNetwork = network(10, 10, 10)

learningRate = 1

for j in range(100):
    inputs = []
    inputs2 = []
    desiredOutputs = []
    
    if (j%50 == 0):
        print(j)

    for i in range(1000):
        inputs.append([(random.randrange(10)/10)])
    inputs = np.array(inputs, dtype=object)

    for i in range(1000):
        desiredOutputs.append([0]*10)
        desiredOutputs[i][9 - int(inputs[i][0]*10)] = 1.0
    desiredOutputs = np.array(desiredOutputs, dtype=object)

    #for i in range(1000):
    #    inputs2.append([0]*10)
    #    inputs2[i][int(inputs[i][0]*10)] = 1.0
    inputs2 = np.array(inputs2, dtype=object)
    
    if (j%10000 == 0):
        learningRate*= 0.1
        
    myNetwork.train(desiredOutputs, desiredOutputs, learningRate)

test = []
test.append([0]*10)
test.append([0]*10)
test[0][1] = 1.0
test[1][8] = 1.0
test = np.array(test, dtype=object)
print(myNetwork.process(test[0]))
print(myNetwork.process(test[1]))