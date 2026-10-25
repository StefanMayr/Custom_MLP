"""Beispiel 4: 1.000 Trainingssamples, 5 Klassen, 10 Test-Predictions.

Bekannte Klassifikationsregel:
    Die Klasse entspricht dem Index des größten der fünf Eingabewerte.
    Beispiel: [0.1, -0.2, 0.9, 0.0, 0.3] -> Klasse 2.

Start:
    python example4.py

Voraussetzung: Das Package Custom_MLP muss importierbar sein.
Dieses Beispiel startet absichtlich mit neuen Gewichten: Alte weights.npz mit
zwei Ausgabeneuronen passen nicht zu einem Modell mit fünf Klassen.
"""

from pathlib import Path
import numpy as np

# Ggf. an die tatsächliche Modulstruktur des Packages anpassen.
#from custom_mlp.core.layer import DenseLayer, ActivationLayer
#from custom_mlp.core.activation_function import ReLu, Softmax
from custom_mlp import Custom_MLP, DenseLayer, ActivationLayer, ReLu, Softmax

RESULTS_DIR = Path(r"D:\PrivatProjekte\Neuronale_Netze\Custom_MLP\results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
NUM_TRAIN = 1000
NUM_TEST = 10
NUM_FEATURES = 5
NUM_CLASSES = 5
rng = np.random.default_rng(SEED)


def make_data(count):
    """Erstellt Eingaben (N, 5, 1) und One-hot-Targets (N, 5, 1)."""
    features = rng.uniform(-1.0, 1.0, size=(count, NUM_FEATURES))
    labels = np.argmax(features, axis=1)
    x = features.reshape(count, NUM_FEATURES, 1)
    y = np.eye(NUM_CLASSES, dtype=float)[labels].reshape(count, NUM_CLASSES, 1)
    return x, y, labels


def main():
    x_data, y_data, train_labels = make_data(NUM_TRAIN)
    x_test, y_test, expected_classes = make_data(NUM_TEST)

    np.savez(
        RESULTS_DIR / "example4_training_data.npz",
        x_data=x_data,
        y_data=y_data,
        labels=train_labels)
    np.savez(
        RESULTS_DIR / "example4_test_data.npz",
        x_data=x_test,
        y_data=y_test,
        labels=expected_classes,)

    # Größere Hidden-Layer als in der 2-Klassen-Version, 5 Softmax-Ausgänge.
    mlp = Custom_MLP(5, [
        DenseLayer(16),
        ActivationLayer(ReLu()),
        DenseLayer(16),
        ActivationLayer(ReLu()),
        DenseLayer(5),
        ActivationLayer(Softmax()),])

    print("Trainingsdaten:", x_data.shape, y_data.shape)
    print("Testdaten:    ", x_test.shape, y_test.shape)
    print("Sollregel: Klasse = Index des größten Eingabewerts")
    print("Trainiere ein neues 5-Klassen-Modell (ohne alte 2-Klassen-Gewichte).")

    # Bei einem Sample müssen 5 Eingaben und 5 Ausgaben vorliegen.
    before = [np.asarray(mlp.predict(x_test[i]), dtype=float).reshape(-1)
              for i in range(NUM_TEST)]

    result = mlp.train(
        x_data,
        y_data,
        epochs=300,
        learning_rate=0.05,
        batch_size=10,  # 1000 ohne Rest teilbar
        loss_function="mse",
        optimizer="gradient",
    )
    print("Trainingsergebnis:", result)

    after = []
    predicted_classes = []
    print("\n10 Vorhersagen auf bisher ungesehenen Samples:")
    print("-" * 95)
    for i in range(NUM_TEST):
        probabilities = np.asarray(mlp.predict(x_test[i]), dtype=float).reshape(-1)
        if probabilities.size != NUM_CLASSES:
            raise ValueError(
                f"predict() sollte {NUM_CLASSES} Softmax-Werte liefern, "
                f"bekam {probabilities.shape}"
            )
        predicted = int(np.argmax(probabilities))
        expected = int(expected_classes[i])
        after.append(probabilities)
        predicted_classes.append(predicted)
        print(
            f"Test {i + 1:2d} | Eingabe: {np.round(x_test[i].reshape(-1), 2)} "
            f"| Soll: {expected} | Ist: {predicted} "
            f"| {'RICHTIG' if predicted == expected else 'FALSCH'}"
        )
        print(f"         Softmax: {np.round(probabilities, 4)}")

    predicted_classes = np.asarray(predicted_classes, dtype=int)
    accuracy = float(np.mean(predicted_classes == expected_classes))
    print("-" * 95)
    print(f"Genauigkeit auf {NUM_TEST} Testsamples: {accuracy:.1%}")
    print("Hinweis: Die Genauigkeit ist nicht garantiert; sie hängt vom Training ab.")

    np.savez(
        RESULTS_DIR / "example4_predictions.npz",
        x_test=x_test,
        expected_classes=expected_classes,
        expected_onehot=y_test,
        probabilities_before=np.stack(before),
        probabilities_after=np.stack(after),
        predicted_classes=predicted_classes,
        accuracy=accuracy,
    )


if __name__ == "__main__":
    main()
