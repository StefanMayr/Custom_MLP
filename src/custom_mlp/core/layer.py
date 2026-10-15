import numpy as np

class InputLayer:
    def __init__(self, input_size):
        self.input_size = input_size
        self.output = None

    def forward(self, x):
        if x.shape[1] != self.input_size:
            raise ValueError(f"Expected input with {self.input_size} features, but got {x.shape[1]}.")
        self.output = x
        return self.output

class HiddenLayer:
    def __init__(self, input_size, output_size, activation_function):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.random.randn(output_size)
        self.activation_function = activation_function
        self.output = None

    def forward(self, x):
        if x.shape[1] != self.input_size:
            raise ValueError(f"Expected input with {self.input_size} features, but got {x.shape[1]}.")
        z = np.dot(x, self.weights) + self.biases
        self.output = self.activation_function.activate(z)
        return self.output

class OutputLayer:
    def __init__(self, input_size, output_size, activation_function):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.random.randn(output_size)
        self.activation_function = activation_function
        self.output = None

    def forward(self, x):
        if x.shape[1] != self.input_size:
            raise ValueError(f"Expected input with {self.input_size} features, but got {x.shape[1]}.")
        z = np.dot(x, self.weights) + self.biases
        self.output = self.activation_function.activate(z)
        return self.output