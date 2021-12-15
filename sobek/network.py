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

    def process(self, _input, __storeValues=False):
        if type(_input) != np.ndarray:
            raise TypeError("The input must be a vector!")
        if _input.size != self.__inputLayerSize:
            raise ValueError("The input vector has the wrong size!")
        #if _input.dtype != np.float64:
        #    raise TypeError("The input vector must contain floats!")

        if (__storeValues):
            self.activations = np.array([])
            self.outputs = np.array([])
        
        for layerWeights, bias in zip(self.__weights, self.__biases):
            _input = np.matmul(_input, layerWeights)
            _input = np.add(_input, bias)

            if (__storeValues):
                print("-------------------")
                print(bias)
                print("-------------------")
                self.activations = np.append(self.activations, _input)
                self.activations[len(self.activations)-1] = np.insert(self.activations[len(self.activations)-1], 0, bias)

            #reLu application
            with np.nditer(_input, op_flags=['readwrite'], flags=['refs_ok']) as layer:
                for neuron in layer:
                    neuron = network.__reLu(neuron)

            #On peut comparer la performance si on recalcul plus tard
            if (__storeValues):
                self.outputs = np.append(self.outputs, _input)
                self.outputs[len(self.outputs)-1] = np.insert(self.outputs[len(self.outputs)-1], 0, 1)
        
        return _input

    def train(self, inputs, desiredOutputs, learningRate):
        ErrorSums = [[0]*(len(layer)+1) for layer in self.__biases]
        for _input, desiredOutput in zip(inputs, desiredOutputs):
            self.__output = self.process(_input, True)
            self.__desiredOutput = desiredOutput
            for layerNumber in range(len(ErrorSums)-1, -1, -1):
                ErrorSums[layerNumber][0] += self.__partialDerivative(layerNumber, 0)
                for neuronNumber in range(1, len(ErrorSums[layerNumber])):
                    print("layer : " + str(layerNumber) + " neuron : " + str(neuronNumber))
                    ErrorSums[layerNumber][neuronNumber] += self.__partialDerivative(layerNumber, neuronNumber)
        for i in range(len(ErrorSums)):
                for j in range(len(ErrorSums[i])):
                    ErrorSums[i][j] = 1 / ErrorSums[i][j]
                    self.__biases[i, j] -= learningRate * ErrorSums[i][j]
        

    def __Error(self, layer, neuron):
        return self.__ErrorFinalLayer(neuron) if (layer == len(self.__weights)-1) else self.__ErrorHiddenLayer(layer, neuron)
        
    def __ErrorFinalLayer(self, neuron):
        print(self.activations)
        return network.__reLu(self.activations[len(self.activations)-1][neuron], True) * (self.__output[neuron] - self.__desiredOutput[neuron])

    def __ErrorHiddenLayer(self, layer, neuron):
        upperLayerLinksSum = 0
        for upperLayerNeuron in range(len(self.__weights[layer+1]-1)):
            #A comparer avec un acces direct au erreurs precalcules
            upperLayerLinksSum += self.__weights[layer+1][upperLayerNeuron][neuron] * self.__Error(layer+1, neuron)
        return network.__reLu(self.activations[layer][neuron], True) * upperLayerLinksSum

    def __partialDerivative(self, layer, neuron):
        return self.__Error(layer, neuron) * self.outputs[layer][neuron]