import numpy as np

class LossFunction:
    def __init__(self, loss_function_type):
        self.loss = None
        self.loss_derivative = None
        self.compute = None
        if loss_function_type == "mse":
            self.compute = self.calculate_mse_lf

    def calculate_mse_lf(self, output: np.ndarray | None = None, output_desired: np.ndarray | None = None):
        self.loss = 0.5 * (output - output_desired)**2
        self.loss_derivative = (output - output_desired)


class Optimizer:
    def __init__(self, method = "gradient", learning_rate = 0.1):
        self.compute = None
        self.learning_rate = learning_rate

        if method == "gradient":
            self.compute = self.gradient_decsent


    def gradient_decsent(self, weights, biases, dW_list, db_list):
        for i in range(len(weights)):
            weights[i] -= self.learning_rate * dW_list[i]
            biases[i] -= self.learning_rate * db_list[i]