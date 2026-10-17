import numpy as np
from custom_mlp.core.layer import DenseLayer, ActivationLayer
from custom_mlp.core.activation_function import ReLu, Sigmoid, Tanh, Softmax
from custom_mlp import Custom_MLP

mlp = Custom_MLP(5, [
    DenseLayer(4),
    ActivationLayer(ReLu()),
    DenseLayer(16),
    ActivationLayer(ReLu()),
    DenseLayer(2),
    ActivationLayer(Softmax())])

output = mlp.predict(np.array([1, 2, 3, 4, 5]))
print("Output:", output)