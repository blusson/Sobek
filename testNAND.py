import numpy as np
import random
from sobek.network import network

myNetwork = network(2, 1)

test = []
test.append(np.zeros(2))
test.append(np.zeros(2))
test.append(np.zeros(2))
test.append(np.zeros(2))
test[1][1] = 1.0
test[2][0] = 1.0
test[3][0] = 1.0
test[3][1] = 1.0

myNetwork.weights = [np.array([[-10.0, -10.0]])]
myNetwork.biases = [np.array([15.0])]
print(myNetwork.weights)
print(myNetwork.biases)

print("0 0 : " + str(myNetwork.process(test[0])) + " == 1 ?")
print("0 1 : " + str(myNetwork.process(test[1])) + " == 1 ?")
print("1 0 : " + str(myNetwork.process(test[2])) + " == 1 ?")
print("1 1 : " + str(myNetwork.process(test[3])) + " == 0 ?")