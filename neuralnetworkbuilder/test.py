import numpy as np

class network:

    def __init__(self, inputLayerSize, *layerSizes):
        self.weights = []
        self.inputLayerSize = inputLayerSize
        self.oldLayerSize = inputLayerSize
        for layerSize in layerSizes:
            self.weights.append( np.random.default_rng(42).random((self.oldLayerSize, layerSize)) )
            self.oldLayerSize = layerSize
        self.biases = [[0]*layerSize for layerSize in layerSizes]
        self.weights = np.array(self.weights)
        self.biases = np.array(self.biases)

    def reLu(value):
        return max(0, value)

    def process(self, input):
        if type(input) != np.ndarray:
            print("non")
        if input.size != self.inputLayerSize:
            print("vite")
        if input.dtype != np.float64:
            print("aaa")
        for layer, bias in zip(self.weights, self.biases):
            print("---------------------")
            print(input)
            print(layer)
            print(bias)
            input = np.matmul(input, layer)
            input = np.add(input, bias)
            with np.nditer(input, op_flags=['readwrite']) as layer:
                for neuron in layer:
                    neuron = network.reLu(neuron)
        return input


test = network(16, 16, 8, 4)

for y in test.weights:
    print(y, end="\n\n")

for y in test.biases:
    print(y, end="\n\n")

print(network.reLu(8))

print(test.process(np.random.default_rng(42).random((16))))