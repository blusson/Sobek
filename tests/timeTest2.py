import random
import numpy as np
import time

weights = np.random.default_rng(42).random((10, 10))
biases = np.random.default_rng(42).random(10)
biases = np.array(biases, dtype=object)

time1 = time.perf_counter()

for k in range(1000):
    _input = []
    for i in range(10):
        _input.append(random.randrange(10))
    _input = np.array(_input, dtype=object)

    for f in range(100):
        _input = np.matmul(_input, weights)
        _input = np.add(_input, biases)

time2 = time.perf_counter()

weights = np.random.default_rng(42).random((11, 10))

time3 = time.perf_counter()

for k in range(1000):
    _input = []
    for i in range(10):
        _input.append(random.randrange(10))
    _input = np.array(_input, dtype=object)

    for f in range(100):
        _input = np.insert(_input, 0, 1, axis=0)
        _input = np.matmul(_input, weights)

time4 = time.perf_counter()

print("Multiplication et addition : " + str(time2-time1) + " secondes")
print("Insertion puis multiplication : " + str(time4-time3) + " secondes")