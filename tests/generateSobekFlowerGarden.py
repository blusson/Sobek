#!/bin/python3
import random
import numpy as np
import math
import pickle

trainPoints = []
trainLabels = []

random.seed(1216513)

for i in range(100):
    x = random.randint(-50, 50)
    y = random.randint(-50, 50)

    distance = math.sqrt(x**2 + y**2)

    if (distance < 10 or 20 < distance < 30):
        trainLabels.append(np.ones(1))
    else :
        trainLabels.append(np.zeros(1))

    x = (x+50)/100
    y = (y+50)/100

    trainPoints.append(np.array([x, y]))

print(trainPoints[1])
print(trainLabels[1])

data = [trainPoints, trainLabels]

with open("flowerGardenData", "wb") as file:
    pickle.dump(data, file)