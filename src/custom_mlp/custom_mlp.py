import numpy as np
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


    def save(self, path):
        data = {
            "input_size": self.input_size,
            "layer_types": np.array([type(layer).__name__ for layer in self.layers])}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, DenseLayer):
                data[f"W_{i}"] = layer.weights
                data[f"b_{i}"] = layer.biases
                data[f"size_{i}"] = layer.size
        np.savez(path, **data)


    @classmethod
    def load(cls, path: str):
        with np.load(path, allow_pickle=False) as data:
            input_size = int(data["input_size"])
            layers = []

            for i, layer_type in enumerate(data["layer_types"]):
                if layer_type == "DenseLayer":
                    size = int(data[f"size_{i}"])
                    layers.append(DenseLayer(size))

                elif layer_type == "ActivationLayer":
                    # Aktivierungsfunktion muss ebenfalls
                    # gespeichert und rekonstruiert werden
                    raise NotImplementedError(
                        "Aktivierungsfunktion rekonstruieren"
                    )

            model = cls(input_size, layers)

            for i, layer in enumerate(model.layers):
                if isinstance(layer, DenseLayer):
                    layer.weights = data[f"W_{i}"].copy()
                    layer.biases = data[f"b_{i}"].copy()

            return model


    def predict(self, input_data):
        self.firstlayer.set_input(input_data)
        current = self.firstlayer
        while current is not None:
            current.forward()
            current = current.next_layer
        return self.lastlayer.output