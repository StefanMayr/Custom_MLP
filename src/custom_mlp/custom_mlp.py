import numpy as np
import json
from pathlib import Path
from .core.layer import Layer, InputLayer, DenseLayer, ActivationLayer
from .core.activation_function import Linear, ReLu, Tanh, Sigmoid, Softmax


class Custom_MLP:
    def __init__(self, input_size: int, layers: list[Layer]):
        self.weights = []
        self.biases = []
        self.firstlayer = InputLayer(input_size)
        layers = [self.firstlayer] + layers

        for i, layer in enumerate(layers):
            if i > 0:
                layer.previous_layer = layers[i - 1]
            if i < len(layers) - 1:
                layer.next_layer = layers[i + 1]
            if isinstance(layer, DenseLayer):
                input_neurons = len(layer.previous_layer)
                weight_index = len(self.weights)
                self.weights.append(np.random.randn(layer.neuron_size, input_neurons) * np.sqrt(1 / input_neurons))
                self.biases.append(np.zeros((layer.neuron_size, 1)))
                layer.initialize(weights=self.weights[weight_index], biases=self.biases[weight_index])
        self.lastlayer = layers[-1]


        def get_architecture(self):
            model = {"formatversion": 1, "inputs": self.firstlayer.output_size}
            current = self.firstlayer
            layerlist = []
            dense_index = 0

            while current is not None:
                if isinstance(current, DenseLayer):
                    layerlist.append({"layertype": "Dense", "neurons": current.neuron_size})
                    model[f"weights_{dense_index}"] = self.weights[dense_index]
                    model[f"biases_{dense_index}"] = self.biases[dense_index]
                    dense_index += 1
                elif isinstance(current, ActivationLayer):
                    layerlist.append({"layertype": "Activation", "type": str(current.activation_function)})
                current = current.next_layer
            model["layers"] = json.dumps(layerlist)
            return model


    def save_model(self):
        model = self.get_architecture()
        np.savez_compressed("D:/PrivatProjekte/Neuronale_Netze/Custom_MLP/results/weights.npz", **model)


    def initialize_model_from_weights(self, model):
        input_size = int(model["inputs"])
        layers_data = model["layers"]
        if isinstance(layers_data, np.ndarray):
            layers_data = layers_data.item()
        if isinstance(layers_data, str):
            layers_data = json.loads(layers_data)

        layers = []
        activations = {
            "Linear": Linear,
            "Sigmoid": Sigmoid,
            "ReLu": ReLu,
            "Tanh": Tanh,
            "Softmax": Softmax
        }

        # 1. Layer rekonstruieren
        for layer_data in layers_data:
            if layer_data["layertype"] == "Dense":
                layers.append(DenseLayer(layer_data["neurons"]))
            elif layer_data["layertype"] == "Activation":
                activation_type = layer_data["type"]
                if activation_type not in activations:
                    raise ValueError(f"Unknown activation: {activation_type}")
                layers.append(ActivationLayer(activations[activation_type]()))
            else:
                raise ValueError(f"Unknown layer type: {layer_data['layertype']}")
        # 2. Konstruktor für zentrale Initialisierung nutzen
        self.__init__(input_size, layers)
        # 3. Gespeicherte Parameter in vorhandene Arrays kopieren
        for i in range(len(self.weights)):
            weights = model[f"weights_{i}"]
            biases = model[f"biases_{i}"]
            if self.weights[i].shape != weights.shape:
                raise ValueError(f"Weight shape mismatch at Dense {i}")
            if self.biases[i].shape != biases.shape:
                raise ValueError(f"Bias shape mismatch at Dense {i}")
            self.weights[i][:] = weights
            self.biases[i][:] = biases


    def load_model(self, model_path):
        with np.load(model_path, allow_pickle=False) as data:
            for key in data.files:
                print(f"{key}: shape={data[key].shape}, dtype={data[key].dtype}")
            model = dict(data.items())
        self.initialize_model_from_weights(model=model)


    def forward_pass(self, input_data):
        self.firstlayer.set_input(input_data)
        current = self.firstlayer
        while current is not None:
            current.forward()
            current = current.next_layer
        return self.lastlayer.output


    def predict(self, input_data):
        return self.forward_pass(input_data=input_data)
    

    def check_folders(self, path):
        return all((path / folder).is_dir() and any((path / folder).iterdir()) for folder in ["train", "test"])


    def backward_pass(self):
        current = self.lastlayer
        pass

    def train(self, data_path, learning_rate=0.1, batch_size=8):
        """
        train_test_path = Path(data_path)
        if self.check_folders(train_test_path) is False:
            print("Ordnersturktur nicht korrekt")
            return
        """
        
        # Routine 
        """
        1. Extract one Example from training data
        2. Calculate Forward pass
        3. Calculate Loss
        4. Backpropagation algo
        5. Gradient Descent
        """

