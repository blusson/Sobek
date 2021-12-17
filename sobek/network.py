import numpy as np
import math

class network:

    def __init__(self, inputLayerSize, *layerSizes):
        if type(inputLayerSize) != int:
            raise TypeError("The input layer size must be an int!")

        self.__weights = []
        self.__inputLayerSize = inputLayerSize
        oldLayerSize = inputLayerSize
        for layerSize in layerSizes:
            self.__weights.append( np.random.random((layerSize, oldLayerSize)) )
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
        return 1/(1+math.exp(-value))

    def process(self, _input, __storeValues=False):
        if type(_input) != np.ndarray:
            raise TypeError("The input must be a vector!")
        if _input.size != self.__inputLayerSize:
            raise ValueError("The input vector has the wrong size!")
        #if _input.dtype != np.float64:
        #    raise TypeError("The input vector must contain floats!")

        if (__storeValues):
            self.activations = []
            self.outputs = []
        
        for layerWeights, bias in zip(self.__weights, self.__biases):
            
            _input = np.matmul(layerWeights, _input)
            _input = np.add(_input, bias)

            if (__storeValues):
                self.activations.append(_input.copy())

            #activation function application
            for neuron in range(len(_input)):
                _input[neuron] = network.__sigmoid(_input[neuron])

            #On peut comparer la performance si on recalcul plus tard
            if (__storeValues):
                self.outputs.append(_input.copy())

        self.activations = np.array(self.activations, dtype=object)
        self.outputs = np.array(self.outputs, dtype=object)
        
        return _input



    def train(self, inputs, desiredOutputs, learningRate):
        errorSumsWeights = [[[0]*len(neuron) for neuron in layer] for layer in self.__weights]
        errorSumsBiases = [[0]*len(layer) for layer in self.__biases]
        self.__errors = [[0]*len(layer) for layer in self.__weights]

        for _input, desiredOutput in zip(inputs, desiredOutputs):

            #rempli self.activations et self.outputs
            self.__output = self.process(_input, True)

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
        self.__weights = np.add(self.__weights, errorSumsWeights)

        errorSumsBiases = np.multiply(errorSumsBiases, -(learningRate/len(inputs)))
        self.__biases = np.add(self.__biases, errorSumsBiases)

        print(self.__biases)
        
        """
        for layerNumber in range(len(errorSumsWeights)):
                for neuronNumber in range(len(errorSumsWeights[layerNumber])):
                    for weightNumber in range(len(errorSumsWeights[layerNumber][neuronNumber])):

                        #Probablement faisable avec une multiplication de matrices
                        errorSumsWeights[layerNumber][neuronNumber][weightNumber] = errorSumsWeights[layerNumber][neuronNumber][weightNumber] / len(inputs)
                        
                        total += errorSumsWeights[layerNumber][neuronNumber][weightNumber]

                        #Probablement faisable avec une somme de matrices
                        self.__weights[layerNumber][neuronNumber][weightNumber] -= learningRate * errorSumsWeights[layerNumber][neuronNumber][weightNumber]

        print("Error : " + str(total))"""

    def __Error(self, layer, neuron):
        if (self.__errors[layer][neuron] == 0 ):
            self.__errors[layer][neuron] = self.__ErrorFinalLayer(neuron) if (layer == len(self.__weights)-1) else self.__ErrorHiddenLayer(layer, neuron)
        return self.__errors[layer][neuron]
        
    def __ErrorFinalLayer(self, neuron):
        return network.__sigmoid(self.activations[len(self.activations)-1][neuron], True) * (self.__output[neuron] - self.__desiredOutput[neuron])

    def __ErrorHiddenLayer(self, layer, neuron):
        upperLayerLinksSum = 0
        #Probablement faisable avec une multiplication de matrices
        for upperLayerNeuron in range(len(self.__weights[layer+1])):
            upperLayerLinksSum += self.__weights[layer+1][upperLayerNeuron][neuron] * self.__errors[layer+1][upperLayerNeuron]
        return network.__sigmoid(self.activations[layer][neuron], True) * upperLayerLinksSum

    def __PartialDerivative(self, layer, neuron, weight):
        return self.__Error(layer, neuron) * self.outputs[layer-1][weight]