import numpy as np
from .core.layer import Layer, InputLayer, DenseLayer, ActivationLayer

class Custom_MLP:
    def __init__(self, input_size: int, layers: list[Layer]):
        # InputLayer automatisch erstellen
        self.firstlayer = InputLayer(input_size)
        layers = [self.firstlayer] + layers

        # Connect all layers
        for i, layer in enumerate(layers):

            if i > 0:
                layer.previous_layer = layers[i - 1]

            if i < len(layers) - 1:
                layer.next_layer = layers[i + 1]

            # DenseLayer initialisieren
            if isinstance(layer, DenseLayer):
                input_neurons = layer.previous_layer.output_size
                layer.initialize(input_neurons)


    def predict(self, input_data):
        self.firstlayer.set_input(input_data)
        head = self.firstlayer
        while head is not None:
            head.forward()
            if head.next_layer is None:
                break
            head = head.next_layer
        return head.output     
