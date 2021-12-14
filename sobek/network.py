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
            return network.__sigmoid(value) * (1 - network.__sigmoid(value))
        return 1/(1+np.exp(-value))

    def process(self, input, storeValues=False):
        if type(input) != np.ndarray:
            raise TypeError("The input must be a vector!")
        if input.size != self.__inputLayerSize:
            raise ValueError("The input vector has the wrong size!")
        if input.dtype != np.float64:
            raise TypeError("The input vector must contain floats!")

        if (storeValues):
            self.activations = []
            self.outputs = []
        
        for layerWeights, bias in zip(self.__weights, self.__biases):
            input = np.matmul(input, layerWeights)
            input = np.add(input, bias)

            if (storeValues):
                self.activations.append(input)

            #reLu application
            with np.nditer(input, op_flags=['readwrite']) as layer:
                for neuron in layer:
                    neuron = network.__reLu(neuron)

            #On peut comparer la performance si on recalcul plus tard
            if (storeValues):
                self.outputs.append(input)
        
        return input

    def train(self, inputs, desiredOutputs):
        for input, desiredOutput in zip(inputs, desiredOutputs):
            self.__output = self.process(input, True)
            self.__desiredOutput = desiredOutput
        #partialDerivatives

    def __Error(self, layer, neuron):
        return self.__ErrorFinalLayer(neuron) if (layer == 1) else self.__ErrorHiddenLayer(layer, neuron)
        
    def __ErrorFinalLayer(self, neuron):
        return network.__reLu(self.activations[len(self.activations)-1][neuron], True) * (self.__output[neuron] - self.__desiredOutput[neuron])

    def __ErrorHiddenLayer(self, layer, neuron):
        upperLayerLinksSum = 0
        for upperLayerNeuron in range(len(self.__weights[layer+1]-1)):
            #A comparer avec un acces direct au erreurs precalcules
            upperLayerLinksSum += self.__weights[layer+1][upperLayerNeuron][neuron] * self.__Error(layer+1, neuron)
        return network.__reLu(self.activations[layer][neuron], True) * upperLayerLinksSum

    def __partialDerivative(self, layer, neuron):
        return self.__Error(layer, neuron) * self.outputs[layer][neuron]