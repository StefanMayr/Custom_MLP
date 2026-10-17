import numpy as np


class Linear:
    def __init__(self):
        pass

    def activate(self, x):
        return x

    def derivative(self, x):
        return np.ones_like(x)


class Sigmoid:
    def __init__(self):
        pass

    def activate(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        sig = self.activate(x)
        return sig * (1 - sig)
    

class ReLu:
    def __init__(self):
        pass

    def activate(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return (x > 0).astype(float)


class Tanh:
    def __init__(self):
        pass

    def activate(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - np.tanh(x) ** 2


class Softmax:
    def __init__(self):
        pass

    def activate(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    def derivative(self, x):
        s = self.activate(x)
        return s * (1 - s)
