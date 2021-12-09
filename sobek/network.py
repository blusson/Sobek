import numpy as np

class network:

    def __init__(self, inputLayerSize, *layerSizes):
        if type(inputLayerSize) != int:
            raise TypeError("The input layer size must be an int!")

        self.__weights = []
        self.__inputLayerSize = inputLayerSize
        oldLayerSize = inputLayerSize
        for layerSize in layerSizes:
            self.__weights.append( np.random.default_rng(42).random((oldLayerSize, layerSize)) )
            oldLayerSize = layerSize
        self.__biases = [[0]*layerSize for layerSize in layerSizes]
        self.__weights = np.array(self.__weights, dtype=object)
        self.__biases = np.array(self.__biases, dtype=object)

    def __reLu(value):
        return max(0, value)

    def process(self, input):
        if type(input) != np.ndarray:
            raise TypeError("The input must be a vector!")
        if input.size != self.__inputLayerSize:
            raise ValueError("The input vector has the wrong size!")
        if input.dtype != np.float64:
            raise TypeError("The input vector must contain floats!")
        
        for layerWeights, bias in zip(self.__weights, self.__biases):
            input = np.matmul(input, layerWeights)
            input = np.add(input, bias)
            #reLu application
            with np.nditer(input, op_flags=['readwrite']) as layer:
                for neuron in layer:
                    neuron = network.__reLu(neuron)
        
        return input

    def train(self, inputs, results):
        