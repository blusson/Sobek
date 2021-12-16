import numpy as np
import random
from sobek.network import network

random.seed()

myNetwork = network(1, 10)

learningRate = 1

for j in range(100000):
    inputs = []
    desiredOutputs = []
    
    if (j%50 == 0):
        print(j)

    for i in range(1000):
        inputs.append([random.randrange(10)])
    inputs = np.array(inputs, dtype=object)

    for i in range(1000):
        desiredOutputs.append([0]*10)
        desiredOutputs[i][9 - inputs[i][0]] = 1.0
    desiredOutputs = np.array(desiredOutputs, dtype=object)
    
    if (j%10000 == 0):
        learningRate*= 0.1
    myNetwork.train(inputs, desiredOutputs, learningRate)

print(myNetwork.process(np.array([0.0], dtype=object)))
print(myNetwork.process(np.array([1.0], dtype=object)))
print(myNetwork.process(np.array([2.0], dtype=object)))
print(myNetwork.process(np.array([3.0], dtype=object)))
print(myNetwork.process(np.array([4.0], dtype=object)))
print(myNetwork.process(np.array([5.0], dtype=object)))
print(myNetwork.process(np.array([6.0], dtype=object)))
print(myNetwork.process(np.array([7.0], dtype=object)))
print(myNetwork.process(np.array([8.0], dtype=object)))
print(myNetwork.process(np.array([9.0], dtype=object)))