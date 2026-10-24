import numpy as np
from abc import ABC, abstractmethod


# ------------------------------------------------
# Base Activation Function
# ------------------------------------------------
class ActivationFunction(ABC):

    @abstractmethod
    def activate(self, x):
        pass

    @abstractmethod
    def derivative(self, x):
        pass

    def backward(self, x, grad_output):
        return self.derivative(x) * grad_output


# ------------------------------------------------
# Linear
# ------------------------------------------------
class Linear(ActivationFunction):

    def activate(self, x):
        return x

    def derivative(self, x):
        return np.ones_like(x)

    def __str__(self):
        return "Linear"


# ------------------------------------------------
# Sigmoid
# ------------------------------------------------
class Sigmoid(ActivationFunction):

    def activate(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        sig = self.activate(x)
        return sig * (1 - sig)

    def __str__(self):
        return "Sigmoid"


# ------------------------------------------------
# ReLU
# ------------------------------------------------
class ReLu(ActivationFunction):

    def activate(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return (x > 0).astype(float)

    def __str__(self):
        return "ReLu"


# ------------------------------------------------
# Tanh
# ------------------------------------------------
class Tanh(ActivationFunction):

    def activate(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def __str__(self):
        return "Tanh"


# ------------------------------------------------
# Softmax
# ------------------------------------------------
class Softmax(ActivationFunction):

    def activate(self, x):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    def derivative(self, x):
        # Vollständige Jacobi-Matrix
        # Für einen einzelnen Spaltenvektor (n, 1)
        s = self.activate(x).reshape(-1)
        return np.diag(s) - np.outer(s, s)

    def backward(self, x, grad_output):
        # Effiziente Berechnung von J.T @ grad_output
        # Ohne die Jacobi-Matrix explizit zu erzeugen
        s = self.activate(x)
        return s * (grad_output - np.sum(s * grad_output, axis=0, keepdims=True))

    def __str__(self):
        return "Softmax"