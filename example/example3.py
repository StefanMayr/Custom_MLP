"""Beispiel 3: Custom_MLP mit 100 künstlichen Klassifikationssamples testen.

Die Klassenregel ist bewusst einfach und vollständig bekannt:
    Klasse 0, falls x[0] + x[1] - x[2] >= 0
    Klasse 1, sonst

Voraussetzungen:
    - Package Custom_MLP ist installiert (z. B. python -m pip install -e .)
    - Eine passende Gewichtsdatei liegt im RESULTS_DIR
    - Die importierten Klassen werden vom Package exportiert;
      gegebenenfalls die Importzeile an die eigene Paketstruktur anpassen.
"""

from pathlib import Path
import numpy as np

# Gegebenenfalls an die tatsächlichen Modulpfade des Packages anpassen.
from custom_mlp.core.layer import DenseLayer, ActivationLayer
from custom_mlp.core.activation_function import ReLu, Softmax
from custom_mlp import Custom_MLP

RESULTS_DIR = Path(r"D:\PrivatProjekte\Neuronale_Netze\Custom_MLP\results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)
NUM_SAMPLES = 100


def make_training_data():
    # Wertebereich [-1, 1], alle Samples besitzen 5 Eingabefeatures.
    features = RNG.uniform(-1.0, 1.0, size=(NUM_SAMPLES, 5))
    scores = features[:, 0] + features[:, 1] - features[:, 2]
    classes = (scores < 0).astype(int)

    # Passend zum bestehenden Netzwerk: (Samples, Features, 1)
    x_data = features.reshape(NUM_SAMPLES, 5, 1)
    # One-hot-Zielwerte passend zu zwei Softmax-Ausgängen.
    y_data = np.eye(2, dtype=float)[classes].reshape(NUM_SAMPLES, 2, 1)
    return x_data, y_data


def expected_class(features):
    return 0 if features[0] + features[1] - features[2] >= 0 else 1


def main():
    weights_path = RESULTS_DIR / "weights.npz"
    if not weights_path.exists():
        alternative = RESULTS_DIR / "weights(2).npz"
        if alternative.exists():
            weights_path = alternative
        else:
            raise FileNotFoundError(
                f"Keine Gewichte gefunden: {weights_path}\n"
                f"Bitte weights.npz (oder weights(2).npz) im results-Ordner ablegen."
            )

    x_data, y_data = make_training_data()
    np.savez(RESULTS_DIR / "example3_training_data.npz", x_data=x_data, y_data=y_data)

    mlp = Custom_MLP(5, [
        DenseLayer(4),
        ActivationLayer(ReLu()),
        DenseLayer(16),
        ActivationLayer(ReLu()),
        DenseLayer(2),
        ActivationLayer(Softmax()),
    ])
    mlp.load_model(str(weights_path))

    # Eingabe und Sollklasse sind absichtlich eindeutig.
    test_sample = np.array([0.9, 0.8, -0.8, 0.2, -0.1], dtype=float)
    target_class = expected_class(test_sample)  # 0
    target_onehot = np.eye(2)[target_class]      # [1., 0.]

    print("Trainingsdaten:", x_data.shape, y_data.shape)
    print("Geladene Gewichte:", weights_path)
    print("Testsample:", test_sample)
    print("Erwartete Klasse:", target_class)
    print("Erwartetes One-hot-Target:", target_onehot)

    # Diese Methode muss ein einzelnes Sample der Form (5, 1) akzeptieren.
    # Falls predict() eine andere Signatur hat, diesen Aufruf anpassen.
    print("Vor dem Training:", mlp.predict(test_sample.reshape(5, 1)))

    result = mlp.train(
        x_data,
        y_data,
        epochs=300,
        learning_rate=0.05,
        batch_size=5,  # 100 ist restlos durch 5 teilbar
        loss_function="mse",
        optimizer="gradient",
    )
    print("Trainingsergebnis:", result)

    prediction = np.asarray(mlp.predict(test_sample.reshape(5, 1)), dtype=float).reshape(-1)
    if prediction.size != 2:
        raise ValueError(f"predict() sollte 2 Softmax-Werte liefern, bekam {prediction.shape}")

    predicted_class = int(np.argmax(prediction))
    print("Nach dem Training – Softmax:", prediction)
    print("Vorhergesagte Klasse:", predicted_class)
    print("Erwartete Klasse:", target_class)
    print("Klasse korrekt:", predicted_class == target_class)

    # Hinweis: Erwartet wird die Klasse [1, 0], nicht exakt die
    # Softmax-Ausgabe [1.0, 0.0]. Softmax liefert Wahrscheinlichkeiten.
    np.savez(
        RESULTS_DIR / "example3_prediction.npz",
        test_sample=test_sample,
        prediction=prediction,
        target_onehot=target_onehot,
        predicted_class=predicted_class,
        expected_class=target_class,
    )


if __name__ == "__main__":
    main()
