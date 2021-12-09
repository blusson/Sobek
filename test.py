import numpy as np
from sobek.network import network

test = network(16, 16, 8, 4)
"""
for y in test.weights:
    print(y, end="\n\n")

for y in test.biases:
    print(y, end="\n\n")"""

#print(network.__reLu(8))

print(test.process(np.random.default_rng(42).random((16))))