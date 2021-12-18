import numpy as np

class network:

    def __init__(self, inputLayerSize, *layerSizes):
        if type(inputLayerSize) != int:
            raise TypeError("The input layer size must be an int!")

        self.weights = []
        self.__inputLayerSize = inputLayerSize
        oldLayerSize = inputLayerSize
        for layerSize in layerSizes:
            self.weights.append( np.random.randn(layerSize, oldLayerSize) )
            oldLayerSize = layerSize
        self.biases = [np.random.randn(layerSize) for layerSize in layerSizes]

    def __reLu(value, derivative=False):
        if (derivative):
            return 0 if (value < 0) else 1
        return max(0, value)

    def __sigmoid(value, derivative=False):
        if (derivative):
            return network.__sigmoid(value) * (1 - network.__sigmoid(value))
        return 1.0/(1.0+np.exp(-value))

    def process(self, _input, __storeValues=False):
        if type(_input) != np.ndarray:
            raise TypeError("The input must be a vector!")
        if _input.size != self.__inputLayerSize:
            raise ValueError("The input vector has the wrong size!")
        if _input.dtype != np.float64:
            print(_input.dtype)
            raise TypeError("The input vector must contain floats!")

        if (__storeValues):
            self.activations = []
            self.outputs = []
            self.outputs.append(_input)
        
        for layerWeights, layerBias in zip(self.weights, self.biases):
            
            _input = np.dot(layerWeights, _input)
            _input = np.add(_input, layerBias)

            if (__storeValues): 
                self.activations.append(_input)

            #activation function application
            #for i in range(len(_input)):
            #    _input[i] = network.__sigmoid(_input)
            _input = network.__sigmoid(_input)

            #On peut comparer la performance si on recalcul plus tard
            if (__storeValues): 
                self.outputs.append(_input)

        return _input



    def train(self, inputs, desiredOutputs, learningRate):
        if (len(inputs) != len(desiredOutputs)):
            raise ValueError("The inputs and desired outputs vectors must have the same amount of data !")

        for _input, desiredOutput in zip(inputs, desiredOutputs):

            errorSumsWeights = [np.zeros(layer.shape) for layer in self.weights]
            errorSumsBiases = [np.zeros(layer.shape) for layer in self.biases]
            self.__errors = [np.zeros(len(layer)) for layer in self.weights]

            #rempli self.activations et self.outputs
            self.process(_input, True)
            self.__desiredOutput = desiredOutput

            #Somme de matrice ?
            for layerNumber in range(len(errorSumsWeights)-1, -1, -1):
                for neuronNumber in range(len(errorSumsWeights[layerNumber])):
                    errorSumsBiases[layerNumber][neuronNumber] += self.__Error(layerNumber, neuronNumber)
                    for weightNumber in range(len(errorSumsWeights[layerNumber][neuronNumber])):
                        #print("layer : " + str(layerNumber) + " neuron : " + str(neuronNumber) + " weight : " + str(weightNumber))
                        errorSumsWeights[layerNumber][neuronNumber][weightNumber] += self.__PartialDerivative(layerNumber, neuronNumber, weightNumber)

        total = 0
        
        
        errorSumsWeights = np.multiply(errorSumsWeights, -(learningRate/len(inputs)))
        self.weights = np.add(self.weights, errorSumsWeights)

        errorSumsBiases = np.multiply(errorSumsBiases, -(learningRate/len(inputs)))
        self.biases = np.add(self.biases, errorSumsBiases)

        #print(self.__biases)
        
        """
        for layerNumber in range(len(errorSumsWeights)):
                for neuronNumber in range(len(errorSumsWeights[layerNumber])):

                    errorSumsBiases[layerNumber][neuronNumber] = errorSumsBiases[layerNumber][neuronNumber] / len(inputs)
                    total += errorSumsBiases[layerNumber][neuronNumber]
                    self.biases[layerNumber][neuronNumber] -= learningRate * errorSumsBiases[layerNumber][neuronNumber]
                    
                    for weightNumber in range(len(errorSumsWeights[layerNumber][neuronNumber])):

                        #Probablement faisable avec une multiplication de matrices
                        errorSumsWeights[layerNumber][neuronNumber][weightNumber] = errorSumsWeights[layerNumber][neuronNumber][weightNumber] / len(inputs)
                        
                        total += errorSumsWeights[layerNumber][neuronNumber][weightNumber]

                        #Probablement faisable avec une somme de matrices
                        self.weights[layerNumber][neuronNumber][weightNumber] -= learningRate * errorSumsWeights[layerNumber][neuronNumber][weightNumber]

        #print("Error : " + str(total))"""

    def __Error(self, layer, neuron):
        if (self.__errors[layer][neuron] == 0 ):
            self.__errors[layer][neuron] = self.__ErrorFinalLayer(neuron) if (layer == len(self.weights)-1) else self.__ErrorHiddenLayer(layer, neuron)
        return self.__errors[layer][neuron]
        
    def __ErrorFinalLayer(self, neuron):
        return network.__sigmoid(self.activations[-1][neuron], derivative=True) * (self.outputs[-1][neuron] - self.__desiredOutput[neuron])

    def __ErrorHiddenLayer(self, layer, neuron):
        upperLayerLinksSum = 0
        #Probablement faisable avec une multiplication de matrices
        for upperLayerNeuron in range(len(self.weights[layer+1])):
            upperLayerLinksSum += self.weights[layer+1][upperLayerNeuron][neuron] * self.__errors[layer+1][upperLayerNeuron]
        return network.__sigmoid(self.activations[layer][neuron], derivative=True) * upperLayerLinksSum

    def __PartialDerivative(self, layer, neuron, weight):
        return self.__Error(layer, neuron) * self.outputs[layer][weight]



    def saveToFile(self, fileName):
        np.savez(fileName, biases=self.biases, weights=self.weights)

    def loadFromFile(self, fileName):
        data = np.load(fileName)
        self.biases = data['biases']
        self.weights = data['weights']