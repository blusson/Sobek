import numpy as np
import random
from sobek.network import network

random.seed()

myNetwork = network(2, 1)

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

for j in range(10000):
    inputs = []
    desiredOutputs = []
    
    if (j%1000 == 0):
        print(j)

    random.shuffle(test)

    for i in range(4):
        if (test[i][0] == 1.0) and (test[i][1] == 1.0):
            result[i][0] = 0.0
        else:
            result[i][0] = 1.0
    
    myNetwork.train(test, result, learningRate)

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

print(myNetwork.weights)
print(myNetwork.biases)
print("0 0 : " + str(myNetwork.process(test[0])) + " == 1 ?")
print("0 1 : " + str(myNetwork.process(test[1])) + " == 1 ?")
print("1 0 : " + str(myNetwork.process(test[2])) + " == 1 ?")
print("1 1 : " + str(myNetwork.process(test[3])) + " == 0 ?")