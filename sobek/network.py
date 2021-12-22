import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle

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
            raise TypeError("The input must be a numpy array!")
        if _input.size != self.__inputLayerSize:
            raise ValueError("The input vector has the wrong size! " + str(_input.size) + " != " + str(self.__inputLayerSize))
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
            _input = network.__sigmoid(_input)

            if (__storeValues): 
                self.outputs.append(_input)

        return _input



    def train(self, inputs, desiredOutputs, learningRate, batchSize, epochs=1, visualize=False):
        if (type(inputs) != list or type(desiredOutputs) != list):
            raise TypeError("The inputs and desired outputs must be lists of numpy arrays !")
        if (len(inputs) != len(desiredOutputs)):
            raise ValueError("The inputs and desired outputs lists must have the same amount of data ! " + str(len(inputs)) + " != " + str(len(desiredOutputs)))
        if (len(inputs) == 0):
            raise ValueError("The list is empty !")
        if (visualize == False):
            if (self.__inputLayerSize != 2):
                raise ValueError("Visualization is only possible for 2 inputs networks")
            if (len(self.weights[-1]) != 1):
                raise ValueError("Visualization is only possible for 1 output networks")

        errorSumsWeights = []
        errorSumsBiases = []

        if (visualize):
            vizualisationData = []
            fig, graph = plt.subplots()

        for epoch in range(epochs):
            randomState = random.getstate()

            random.shuffle(inputs)

            random.setstate(randomState)

            random.shuffle(desiredOutputs)

            if (visualize and epoch%10 == 0):
                vizualisationFrame = np.empty((30, 30))
                for x in range(30):
                    for y in range(30):
                        vizualisationFrame[x][y] = self.process(np.array([float(x), float(y)]))
                vizualisationData.append([graph.imshow(vizualisationFrame, animated=True)])

            inputBatches = [inputs[j:j+batchSize] for j in range(0, len(inputs), batchSize)]
            desiredOutputsBatches = [desiredOutputs[j:j+batchSize] for j in range(0, len(inputs), batchSize)]

            for inputBatch, desiredOutputsBatch in zip(inputBatches, desiredOutputsBatches):
                    
                for _input, desiredOutput in zip(inputBatch, desiredOutputsBatch):

                    errorSumsWeights = [np.zeros(layer.shape) for layer in self.weights]
                    errorSumsBiases = [np.zeros(layer.shape) for layer in self.biases]
                    self.__errors = [np.zeros(len(layer)) for layer in self.weights]

                    #Rempli self.activations et self.outputs
                    self.process(_input, True)

                    self.__desiredOutput = desiredOutput

                    for layerNumber in range(len(errorSumsWeights)-1, -1, -1):
                        for neuronNumber in range(len(errorSumsWeights[layerNumber])):
                            errorSumsBiases[layerNumber][neuronNumber] += self.__Error(layerNumber, neuronNumber)
                            errorSumsWeights[layerNumber][neuronNumber] = np.dot(errorSumsBiases[layerNumber][neuronNumber],self.outputs[layerNumber])

                total = 0
                
                for layerNumber in range(len(errorSumsWeights)):
                    errorSumsWeights[layerNumber] = np.multiply(errorSumsWeights[layerNumber], -(learningRate/len(inputBatch)))
                    self.weights[layerNumber] = np.add(self.weights[layerNumber], errorSumsWeights[layerNumber])

                    errorSumsBiases[layerNumber] = np.multiply(errorSumsBiases[layerNumber], -(learningRate/len(inputBatch)))
                    self.biases[layerNumber] = np.add(self.biases[layerNumber], errorSumsBiases[layerNumber])

        if (visualize):
            ani = animation.ArtistAnimation(fig, vizualisationData, interval=100)
            plt.show()

    def __Error(self, layer, neuron):
        if (self.__errors[layer][neuron] == 0 ):
            self.__errors[layer][neuron] = self.__ErrorFinalLayer(neuron) if (layer == len(self.weights)-1) else self.__ErrorHiddenLayer(layer, neuron)
        return self.__errors[layer][neuron]
        
    def __ErrorFinalLayer(self, neuron):
        return network.__sigmoid(self.activations[-1][neuron], derivative=True) * (self.outputs[-1][neuron] - self.__desiredOutput[neuron])

    def __ErrorHiddenLayer(self, layer, neuron):
        upperLayerLinksSum = 0
        for upperLayerNeuron in range(len(self.weights[layer+1])):
            upperLayerLinksSum += self.weights[layer+1][upperLayerNeuron][neuron] * self.__errors[layer+1][upperLayerNeuron]
        return network.__sigmoid(self.activations[layer][neuron], derivative=True) * upperLayerLinksSum

    def accuracy(self, inputs, desiredOutputs):
        if (type(inputs) != list or type(desiredOutputs) != list):
            raise TypeError("The inputs and desired outputs must be lists of numpy arrays !")
        if (len(inputs) != len(desiredOutputs)):
            raise ValueError("The inputs and desired outputs lists must have the same amount of data !")
        if (len(inputs) == 0):
            raise ValueError("The list is empty !")
        sum = 0
        for i in range(len(desiredOutputs)):
            if (np.argmax(desiredOutputs[i]) == np.argmax(self.process(inputs[i]))):
                sum += 1
        return sum/len(desiredOutputs)


    def saveToFile(self, fileName):
        with open(fileName, "wb") as file:
            pickle.dump(self, file)

    def loadFromFile(self, fileName):
        with open(fileName, "rb") as file:
            fromNetwork = pickle.load(file)
            self.weights = fromNetwork.weights
            self.biases = fromNetwork.biases
            self.__inputLayerSize = fromNetwork.__inputLayerSize

    def networkFromFile(fileName):
        with open(fileName, "rb") as file:
            return pickle.load(file)