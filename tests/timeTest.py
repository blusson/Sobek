import random
import numpy as np

inputs = []

for i in range(10000000):
    inputs.append([random.randrange(10)])
inputs = np.array(inputs, dtype=object)

inputs = np.insert(inputs, 0, 1, axis=1)

print(inputs)