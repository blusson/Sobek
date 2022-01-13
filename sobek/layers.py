import numpy as np

class layer:
    def __init__(self, neurons, activationFunction)
        self.neurons = neurons
        self.activationFunction = activationFunction

    def process(_input, __storeValues=False)

class dense(layer):
    def process(_input, __storeValues=False):
        
        _input = np.dot(layerWeights, _input)
        _input = np.add(_input, layerBias)

        if (__storeValues):
            self.activation = _input

        _input = self.activationFunction.applyTo(_input)

        if (__storeValues):
            self.output = _input

        return _input

class convolution(layer):
    pass

class flatten(layer):
    pass