from .core.layer import Layer, InputLayer, DenseLayer


class Custom_MLP:
    def __init__(self, input_size: int, layers: list[Layer]):
        self.firstlayer = InputLayer(input_size)
        layers = [self.firstlayer] + layers

        for i, layer in enumerate(layers):
            if i > 0:
                layer.previous_layer = layers[i - 1]
            if i < len(layers) - 1:
                layer.next_layer = layers[i + 1]
            if isinstance(layer, DenseLayer):
                input_neurons = len(layer.previous_layer)
                layer.initialize(input_neurons)
        self.lastlayer = layers[-1]


    def predict(self, input_data):
        self.firstlayer.set_input(input_data)
        current = self.firstlayer
        while current is not None:
            current.forward()
            current = current.next_layer
        return self.lastlayer.output