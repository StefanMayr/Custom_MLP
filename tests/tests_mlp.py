import numpy as np
import pytest

from custom_mlp import (Custom_MLP, DenseLayer, ActivationLayer, ReLu, Sigmoid,)

def test_binary_classification():
    mlp = Custom_MLP(
        input_size=3,
        layers=[
            DenseLayer(8),
            ActivationLayer(ReLu()),
            DenseLayer(1),
            ActivationLayer(Sigmoid())
        ]
    )

    x = np.array([
        [1.0],
        [2.0],
        [3.0]
    ])

    prediction = mlp.predict(x)

    assert prediction.shape == (1, 1)
    assert np.all(prediction >= 0)
    assert np.all(prediction <= 1)


def test_multiclass_classification():
    mlp = Custom_MLP(
        input_size=3,
        layers=[
            DenseLayer(8),
            ActivationLayer(ReLu()),
            DenseLayer(4),
            ActivationLayer(Softmax())
        ]
    )

    x = np.array([[1.0], [2.0], [3.0]])
    prediction = mlp.predict(x)
    assert prediction.shape == (4, 1)
    # Softmax muss sich zu 1 summieren
    assert np.isclose(np.sum(prediction), 1.0)
    assert np.all(prediction >= 0)
    assert np.all(prediction <= 1)


def test_invalid_input_shape():
    mlp = Custom_MLP(
        input_size=3,
        layers=[
            DenseLayer(1),
            ActivationLayer(Sigmoid())
        ]
    )

    x = np.array([[1.0], [2.0]])

    with pytest.raises(ValueError):
        mlp.predict(x)