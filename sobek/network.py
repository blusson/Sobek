import numpy as np

class network:

    def __init__(self, inputLayerSize, *layerSizes):
        if type(inputLayerSize) != int:
            raise TypeError("The input layer size must be an int!")

        self.weights = []
        self.inputLayerSize = inputLayerSize
        self.oldLayerSize = inputLayerSize
        for layerSize in layerSizes:
            self.weights.append( np.random.default_rng(42).random((self.oldLayerSize, layerSize)) )
            self.oldLayerSize = layerSize
        self.biases = [[0]*layerSize for layerSize in layerSizes]
        self.weights = np.array(self.weights, dtype=object)
        self.biases = np.array(self.biases, dtype=object)

    def reLu(value):
        return max(0, value)

    def process(self, input):
        if type(input) != np.ndarray:
            raise TypeError("The input must be a vector!")
        if input.size != self.inputLayerSize:
            raise ValueError("The input vector has the wrong size!")
        if input.dtype != np.float64:
            raise TypeError("The input vector must contain floats!")
        
        for layerWeights, bias in zip(self.weights, self.biases):
            input = np.matmul(input, layerWeights)
            input = np.add(input, bias)
            #reLu application
            with np.nditer(input, op_flags=['readwrite']) as layer:
                for neuron in layer:
                    neuron = network.reLu(neuron)
        
        return input