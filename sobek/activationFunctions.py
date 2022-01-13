class activationFunction:
    def applyTo(value):
        pass

    def applyDerivateTo(value):
        pass

class sigmoid(activationFunction):
    def applyTo(value):
        return 1.0/(1.0+np.exp(-value))

    def applyDerivateTo(value):
        return sigmoid.applyTo(value) * (1 - sigmoid.applyTo(value))

class reLu(activationFunction):
    def applyTo(value):
        return max(0, value)

    def applyDerivateTo(value):
        return 0 if (value < 0) else 1

class softMax(activationFunction):
    pass