import numpy as np
import json
from .core.layer import Layer, InputLayer, DenseLayer, ActivationLayer
from .core.activation_function import Linear, ReLu, Tanh, Sigmoid, Softmax


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


    def get_architecture(self):
        model = {}

        model["formatversion"] = 1
        model["inputs"] = self.firstlayer.output_size

        current = self.firstlayer
        layerlist = []
        dense_index = 0

        while current is not None:
            if isinstance(current, DenseLayer):
                layerlist.append({"layertype": "Dense", "neurons": current.weights.shape[0]})
                # Gewichte und Biases einzeln speichern
                model[f"weights_{dense_index}"] = current.weights
                model[f"biases_{dense_index}"] = current.biases
                dense_index += 1
            elif isinstance(current, ActivationLayer):
                layerlist.append({"layertype": "Activation", "type": str(current.activation_function)})
            current = current.next_layer
        # Liste von Dictionaries als JSON-String speichern
        model["layers"] = json.dumps(layerlist)
        return model


    def save_model(self):
        model = self.get_architecture()
        np.savez_compressed("D:/PrivatProjekte/Neuronale_Netze/Custom_MLP/results/weights.npz", **model)


    
    def initialize_model_from_weights(self, model):
        import json
        import numpy as np

        # InputLayer erstellen
        self.firstlayer = InputLayer(int(model["inputs"]))

        # Architektur auslesen
        layers = model["layers"]

        # Falls aus NPZ geladen: 0D-Array zu String
        if isinstance(layers, np.ndarray):
            layers = layers.item()

        # JSON-String zu Python-Liste
        if isinstance(layers, str):
            layers = json.loads(layers)

        layer_mlp_list = []
        dense_index = 0

        # Alle Layer erstellen
        for layer in layers:

            if layer["layertype"] == "Dense":
                denselayer = DenseLayer(layer["neurons"])
                layer_mlp_list.append(denselayer)

            elif layer["layertype"] == "Activation":
                activation_type = layer["type"]

                if activation_type == "Linear":
                    layer_mlp_list.append(ActivationLayer(Linear()))

                elif activation_type == "Sigmoid":
                    layer_mlp_list.append(ActivationLayer(Sigmoid()))

                elif activation_type == "ReLu":
                    layer_mlp_list.append(ActivationLayer(ReLu()))

                elif activation_type == "Tanh":
                    layer_mlp_list.append(ActivationLayer(Tanh()))

                elif activation_type == "Softmax":
                    layer_mlp_list.append(ActivationLayer(Softmax()))

                else:
                    raise ValueError(
                        f"Unknown activation: {activation_type}"
                    )

            else:
                raise ValueError(
                    f"Unknown layer type: {layer['layertype']}"
                )

        # InputLayer und restliche Layer zusammenführen
        all_layers = [self.firstlayer] + layer_mlp_list

        # Layer miteinander verbinden
        for i, layer in enumerate(all_layers):

            layer.previous_layer = (
                all_layers[i - 1] if i > 0 else None
            )

            layer.next_layer = (
                all_layers[i + 1]
                if i < len(all_layers) - 1 else None
            )

        # Dense-Layer initialisieren und Gewichte laden
        for layer in all_layers:

            if isinstance(layer, DenseLayer):

                input_neurons = len(layer.previous_layer)

                # Layer initialisieren
                layer.initialize(input_neurons)

                # Gespeicherte Gewichte und Biases laden
                weights = model[f"weights_{dense_index}"]
                biases = model[f"biases_{dense_index}"]

                # Dimensionen überprüfen
                if layer.weights.shape != weights.shape:
                    raise ValueError(
                        f"Weight shape mismatch at Dense {dense_index}: "
                        f"expected {layer.weights.shape}, "
                        f"got {weights.shape}"
                    )

                if layer.biases.shape != biases.shape:
                    raise ValueError(
                        f"Bias shape mismatch at Dense {dense_index}: "
                        f"expected {layer.biases.shape}, "
                        f"got {biases.shape}"
                    )

                # Gespeicherte Parameter übernehmen
                layer.weights = weights.copy()
                layer.biases = biases.copy()

                dense_index += 1

        # Letzten Layer setzen
        self.lastlayer = all_layers[-1]


    def load_model(self, model_path):
        with np.load(model_path, allow_pickle=False) as data:
            for key in data.files:
                print(f"{key}: shape={data[key].shape}, dtype={data[key].dtype}")
            model = dict(data.items())
        self.initialize_model_from_weights(model=model)


    def predict(self, input_data):
        self.firstlayer.set_input(input_data)
        current = self.firstlayer
        while current is not None:
            current.forward()
            current = current.next_layer
        return self.lastlayer.output