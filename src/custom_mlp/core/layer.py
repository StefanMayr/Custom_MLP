import numpy as np
from .activation_function import Linear, Sigmoid, ReLu, Tanh, Softmax

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
            raise ValueError(f"Expected input shape ({self.output_size}, 1), got {input_data.shape}")
        self.input_data = input_data

    def forward(self):
        self.output = self.input_data

    def backward(self):
        return None, None

    def __len__(self):
        return self.output_size


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

    def initialize(self, weights, biases):
        self.weights = weights
        self.biases = biases
        #self.weights = np.random.randn(self.neuron_size, input_size) * np.sqrt(1 / input_size)
        #self.biases = np.zeros((self.neuron_size, 1))

    def forward(self):
        self.output = self.weights @ self.previous_layer.output + self.biases
        #print(f"Local_weights = {self.weights}")
        #print(f"denselayer_output = {self.output}")

    def backward(self):
        dW = self.next_layer.delta @ self.previous_layer.output.T
        db = self.next_layer.delta
        self.delta = self.weights.T @ self.next_layer.delta
        return dW, db

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

    def get_bias(self):
        return self.biases

    def set_biases(self, biases):
        self.biases = biases

    def __len__(self):
        return self.neuron_size


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
        self.delta = None


    def forward_normal(self):
        self.output = self.activation_function.activate(self.previous_layer.output)
        #print(f"actilayer_output = {self.output}")

    def forward_softmax(self):
        z = self.previous_layer.output
        exp_z = np.exp(z - np.max(z))
        self.output = exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def backward(self):
        self.delta = self.activation_function.backward(self.previous_layer.output, self.next_layer.delta)
        return None, None

    def backward_last(self, loss_gradient):
        self.delta = self.activation_function.backward(self.previous_layer.output, loss_gradient)
        return None, None

    def __len__(self):
        return len(self.previous_layer)
     