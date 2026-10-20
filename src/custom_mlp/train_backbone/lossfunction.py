import numpy as np

class LossFunction:
    def __init__(self, loss_function_type):
        self.loss = None
        self.loss_derivative = None
        self.lossfunction = None
        if loss_function_type == "Quadratic_Loss":
            self.lossfunction = self.calculate_quadratic_loss_fuction

    def calculate_quadratic_loss_fuction(self, output: np.ndarray | None = None, output_desired: np.ndarray | None = None):
        self-self.loss = 0.5 * (output - output_desired)**2
        self.loss_derivative = (output - output_desired)