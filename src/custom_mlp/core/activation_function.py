import numpy as np

class ActivationFunction:
    def __init__(self, function_type):
        if function_type not in ['sigmoid', 'relu', 'tanh']:
            raise ValueError(f"Unsupported activation function: {function_type}")
        if function_type == 'sigmoid':
            self.derivative = lambda x: self.activate(x) * (1 - self.activate(x))
        self.function_type = function_type

    #-------------------------------------
    # Activation Functions
    #-------------------------------------
    def activate_relu(self, x):
        return np.maximum(0, x)

    def activate_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def activate_tanh(self, x):
        return np.tanh(x)

    #-------------------------------------
    # Derivative Functions
    #-------------------------------------
    def derivative_relu(self, x):
        return (x > 0).astype(float)

    def derivative_sigmoid(self, x):
        return np.exp(-x) / ((1 + np.exp(-x))**2)

    def derivative_tanh(self, x):
        return (1 - np.tanh(x)**2)