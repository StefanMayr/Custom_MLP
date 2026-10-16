import numpy as np
from .activation_function import Linear, Sigmoid, ReLu, Tanh

#------------------------------------------------
# Layer Classes
class Layer:
    def __init__(self):
        self.previous_layer = None
        self.next_layer = None

    def forward(self, x):
        raise NotImplementedError

    def backward(self, gradient):
        raise NotImplementedError


#------------------------------------------------
# Input Layer
class InputLayer(Layer):
    def __init__(self, input_size):
        super().__init__()
        self.output_size = input_size
        self.input_data = None
        self.output = None

    def set_input(self, input_data):
        input_data = np.asarray(input_data, dtype=float)
        if input_data.shape == (self.output_size,):
            input_data = input_data.reshape(-1, 1)
        if input_data.shape != (self.output_size, 1):
            raise ValueError(f"Expected input shape ({self.output_size}, 1), got {x.shape}")
        self.input_data = input_data

    def forward(self):
        self.output = self.input_data


#------------------------------------------------
# Dense Layer
class DenseLayer(Layer):
    def __init__(self, neuron_size):
        super().__init__()
        self.neuron_size = neuron_size
        self.weights = None
        self.biases = None

        # z-value
        self.output = None
        self.delta = None

    def initialize(self, input_size):
        self.weights = np.random.randn(self.neuron_size, input_size) * np.sqrt(1 / input_size)
        self.biases = np.zeros((self.neuron_size, 1))

    def forward(self):
        self.z = self.weights @ self.previous_layer.output + self.biases


#------------------------------------------------
# Activation Layer
class ActivationLayer(Layer):
    def __init__(self, activation_function):
        super().__init__()

        if activation_function == "softmax":
            self.forward = self.forward_softmax
        else:
            self.forward = self.forward_normal

        self.activation_function = activation_function
        self.output = None

    def forward_normal(self):
        self.output = self.activation_function.activate(self.previous_layer.output)

    def forward_softmax(self):
        z = self.previous_layer.output
        exp_z = np.exp(z - np.max(z))
        self.output = exp_z / np.sum(exp_z, axis=0, keepdims=True)
        

