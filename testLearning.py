import numpy as np
import random
from sobek.network import network

random.seed()

myNetwork = network(1, 8, 8, 10)

for j in range(5):
    inputs = []
    desiredOutputs = []

    for i in range(1000):
        inputs.append([random.randrange(10)])
    inputs = np.array(inputs, dtype=object)

    for i in range(1000):
        desiredOutputs.append([0]*10)
        desiredOutputs[i][9 - inputs[i][0]] = 1
    desiredOutputs = np.array(desiredOutputs, dtype=object)

    myNetwork.train(inputs, desiredOutputs, 0.1)

print(myNetwork.process(np.array([8.0], dtype=object)))
print(myNetwork.process(np.array([7.0], dtype=object)))