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

    def __reLu(value, derivative=False):
        if (derivative):
            return 0 if (value == 0) else 1
        return max(0, value)

    def __sigmoid(value, derivative=False):
        if (derivative):
            return __sigmoid(value) * (1 - __sigmoid(value))
        return 1/(1+np.exp(-value))

    def process(self, input, storeValues=False):
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
        self.__outputs = 1
        #for j in range(1,):
        #partialDerivatives

    def __Error(layer, output, desiredOutput):
        return __ErrorFinalLayerFromValue() if (layer == 1)
        
    def __ErrorFinalLayer(self, neuron):
        return __reLu(value, true) * (output - desiredOutput)