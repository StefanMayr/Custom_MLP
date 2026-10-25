import numpy as np
import json
from pathlib import Path
from .core.layer import Layer, InputLayer, DenseLayer, ActivationLayer
from .core.activation_function import Linear, ReLu, Tanh, Sigmoid, Softmax
from .train_backbone.training import LossFunction, Optimizer


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

    def print_weights(self):
        print(self.weights)

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
        activations = {"Linear": Linear, "Sigmoid": Sigmoid, "ReLu": ReLu, "Tanh": Tanh, "Softmax": Softmax}
        layers = []
        # 1. Architektur rekonstruieren
        for layer_data in layers_data:

            if layer_data["layertype"] == "Dense":
                layers.append(DenseLayer(int(layer_data["neurons"])))
            elif layer_data["layertype"] == "Activation":
                activation_type = layer_data["type"]
                layers.append(ActivationLayer(activations[activation_type]()))
        # 2. Neues Netzwerk erstellen
        self.__init__(input_size, layers)

        # 3. Gewichte laden und Referenzen neu setzen
        current = self.firstlayer
        dense_index = 0

        while current is not None:
            if isinstance(current, DenseLayer):
                weights = np.asarray(model[f"weights_{dense_index}"])
                biases = np.asarray(model[f"biases_{dense_index}"])
                # Dimensionen überprüfen
                if self.weights[dense_index].shape != weights.shape:
                    raise ValueError(f"Dense {dense_index}: Expected {self.weights[dense_index].shape}, got {weights.shape}")
                if self.biases[dense_index].shape != biases.shape:
                    raise ValueError(f"Dense {dense_index}: Expected {self.biases[dense_index].shape}, got {biases.shape}")
                # Zentrale Arrays ersetzen
                self.weights[dense_index] = weights.copy()
                self.biases[dense_index] = biases.copy()

                # Referenzen des DenseLayers neu setzen
                current.weights = self.weights[dense_index]
                current.biases = self.biases[dense_index]
                dense_index += 1
            current = current.next_layer


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

    def validate_y_data(self, y_data, num_samples):
        y_data = np.asarray(y_data)

        # Anzahl der Output-Neuronen bestimmen
        current = self.lastlayer
        while current is not None and not isinstance(current, DenseLayer):
            current = current.previous_layer
        if current is None:
            raise ValueError("No DenseLayer found")
        output_neurons = current.neuron_size

        # Einzelne Zielwerte / 1D-Vektoren behandeln
        if y_data.ndim == 0:
            y_data = y_data.reshape(1, 1)
        elif y_data.ndim == 1:
            if num_samples == 1:
                y_data = y_data.reshape(-1, 1)
            elif output_neurons == 1 and y_data.size == num_samples:
                y_data = y_data.reshape(1, -1)
            else:
                raise ValueError("Ambiguous 1D target data")
        # Korrekte Shape überprüfen
        expected_shape = (output_neurons, num_samples)
        if y_data.shape != expected_shape:
            raise ValueError(f"Invalid y_data shape: {y_data.shape}, expected {expected_shape}")
        return y_data

    def backward_pass(self):
        current = self.lastlayer
        while current is not None:
            current.backward()
            current = current.previous_layer

    def train(self, x_data, y_data, epochs=1, learning_rate=0.1, batch_size=8, loss_function = "mse", optimizer = "gradient"):
        self.optimizer = Optimizer(optimizer, learning_rate=learning_rate)
        self.loss_function = LossFunction(loss_function_type=loss_function)

        if x_data.shape[0] != y_data.shape[0]:
            print(f"x_data length does not match y_data length")
            return None
        #y_data = y_data.reshape(-1, 1)
        x_data = np.asarray(x_data)
        if x_data.ndim == 1:
            x_data = x_data.reshape(-1, 1)
        if x_data.ndim != 3 or x_data.shape[1] != self.firstlayer.output_size:
            raise ValueError("Invalid x_data shape")
        num_of_training = x_data.shape[1]
        
        #num_of_training = x_data.shape[0]
        num_of_training = int(x_data.shape[0]/batch_size) # Öeave the rest 

        # Loop for epochs (How many times on the same dataset)
        for epoch in range(1, epochs+1):
            for iteration in range(num_of_training):
                # Prototype => one batch (currently all)
                dW_batch_list = []
                db_batch_list = []

                for batch in range(batch_size):
                    dW_network_list = []
                    db_network_list = []
                    # 1. Make forward pass
                    self.forward_pass(x_data[batch+iteration*batch_size].reshape(-1, 1))

                    # 2. Calculate Loss function
                    #loss = self.lastlayer.output - y_data[iteration].reshape(-1,1)
                    self.loss_function.compute(self.lastlayer.output, y_data[batch + iteration * batch_size].reshape(-1,1))  

                    # 3.backward pass
                    self.lastlayer.backward_last(self.loss_function.loss_derivative)
                    current = self.lastlayer.previous_layer
                    while current is not None:
                        dW, db = current.backward()
                        if dW is not None:
                            dW_network_list.insert(0, dW)
                        if db is not None:
                            db_network_list.insert(0, db)
                        current = current.previous_layer
                    # 4. save all batches
                    dW_batch_list.append(dW_network_list)
                    db_batch_list.append(db_network_list)
                # 5. Averaging over all batches
                dW_averaged = [np.mean(arrays, axis=0) for arrays in zip(*dW_batch_list)]
                db_averaged = [np.mean(arrays, axis=0) for arrays in zip(*db_batch_list)]

                # 4. Gradient descent
                self.optimizer.compute(self.weights, self.biases, dW_averaged, db_averaged)
        return True