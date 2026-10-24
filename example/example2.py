"""Ein Backpropagation-Schritt und Vergleich mit gespeicherten Referenzgradienten.

Erwartet in <Projekt>/results:
    weights.npz, trainingsdaten.npz, expected_gradients.npz

Unterstützte Rückgabeformate von train():
    (dW, db) mit je einem NumPy-Array für EINEN Dense-Layer
    (dW, db) mit je einer Liste/Tuple aus Arrays für ALLE Dense-Layer

Voraussetzung: train() gibt Gradienten VOR einem möglichen Optimizer-Update
zurück und verwendet dieselbe Loss-Funktion wie die Referenzdaten.
"""
from pathlib import Path
import numpy as np

from custom_mlp.core.layer import DenseLayer, ActivationLayer
from custom_mlp.core.activation_function import ReLu, Softmax
from custom_mlp import Custom_MLP

np.set_printoptions(precision=10, suppress=True)

# Beispiel liegt üblicherweise in <Projekt>/example/example.py.
# Die drei NPZ-Dateien liegen dann unter <Projekt>/results/.
PROJECT_DIR = Path(__file__).resolve().parent.parent
path = PROJECT_DIR / "results"
if not path.is_dir():
    # Alternativ liegt example.py direkt im Projektverzeichnis.
    path = Path(__file__).resolve().parent / "results"

for name in ("weights.npz", "trainingsdaten.npz", "expected_gradients.npz"):
    file = path / name
    if not file.is_file():        
        raise FileNotFoundError(f"Testdatei fehlt: {file}")

# Diese anfängliche Architektur wird durch load_model() ersetzt.
mlp = Custom_MLP(5, [
    DenseLayer(4),
    ActivationLayer(ReLu()),
    DenseLayer(16),
    ActivationLayer(ReLu()),
    DenseLayer(2),
    ActivationLayer(Softmax()),
])
mlp.load_model(str(path / "weights.npz"))
#mlp.print_weights()

with np.load(path / "trainingsdaten.npz") as data:
    x_data = data["x_data"]
    y_data = data["y_data"]

with np.load(path / "expected_gradients.npz") as data:
    expected = {key: data[key].copy() for key in data.files}

print("x_data:", x_data)
print("y_data:", y_data)

# EIN Trainingsbeispiel, EIN Backpropagation-Schritt.
# Falls deine train()-Methode andere Argumente verlangt, nur hier anpassen.
result = mlp.train(x_data, y_data)

if not isinstance(result, (tuple, list)) or len(result) != 2:
    raise TypeError(
        "train() muss (dW, db) zurückgeben. "
        f"Tatsächlicher Rückgabetyp: {type(result).__name__}, "
        f"Wert: {repr(result)[:300]}")

dW, db = result
#print(dW)


def as_gradient_list(value, name):
    """Ein einzelnes Array oder eine Liste von Gradienten vereinheitlichen."""
    if isinstance(value, np.ndarray):
        return [value]
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"{name} ist leer.")
        # Auch eine Python-Liste mit einzelnen Zahlen / verschachtelten Zahlen
        # kann ein einzelnes Array sein; normalerweise werden aber Arrays übergeben.
        if all(isinstance(v, (np.ndarray, list, tuple)) for v in value):
            return [np.asarray(v) for v in value]
        return [np.asarray(value)]
    raise TypeError(f"{name} muss ein Array oder eine Sequenz aus Arrays sein, nicht {type(value).__name__}.")


dW_list = as_gradient_list(dW, "dW")
db_list = as_gradient_list(db, "db")
if len(dW_list) != len(db_list):
    raise ValueError(f"Unterschiedliche Anzahl: {len(dW_list)} dW, {len(db_list)} db")

layer_ids = sorted(int(k.split("_")[1]) for k in expected if k.startswith("dW_"))
if not layer_ids:
    raise ValueError("Keine Referenzgradienten dW_0, dW_1, ... gefunden.")

# Bei vollständiger Gradientenliste: Reihenfolge Dense 0, Dense 1, ...
if len(dW_list) == len(layer_ids):
    assignments = list(zip(layer_ids, dW_list, db_list))
# Bei einem einzigen Gradientenpaar: passenden Dense-Layer anhand beider Shapes finden.
elif len(dW_list) == 1:
    w, b = np.asarray(dW_list[0]), np.asarray(db_list[0])
    matches = [i for i in layer_ids
               if w.shape == expected[f"dW_{i}"].shape
               and b.shape == expected[f"db_{i}"].shape]
    if len(matches) != 1:
        raise ValueError(
            "Einzelnen Gradienten konnte ich keinem Layer eindeutig zuordnen. "
            f"dW.shape={w.shape}, db.shape={b.shape}; Treffer={matches}. "
            "Falls nötig, Layerindex für den Vergleich explizit setzen."
        )
    assignments = [(matches[0], w, b)]
else:
    raise ValueError(
        f"train() gab {len(dW_list)} Gradientenpaare zurück, Referenz hat "
        f"{len(layer_ids)} Dense-Layer. Für eine Teilmenge ist die Zuordnung "
        "nicht eindeutig; gib die Layerindizes explizit an."
    )

all_ok = True
for i, actual_w, actual_b in assignments:
    actual_w, actual_b = np.asarray(actual_w), np.asarray(actual_b)
    reference_w = expected[f"dW_{i}"]
    reference_b = expected[f"db_{i}"]

    w_shape_ok = actual_w.shape == reference_w.shape
    b_shape_ok = actual_b.shape == reference_b.shape
    w_ok = w_shape_ok and np.allclose(actual_w, reference_w, rtol=1e-6, atol=1e-9)
    b_ok = b_shape_ok and np.allclose(actual_b, reference_b, rtol=1e-6, atol=1e-9)
    all_ok &= bool(w_ok and b_ok)

    print(f"\nDense {i}")
    print(f"dW: {'OK' if w_ok else 'FEHLER'} (Ist {actual_w.shape}, Soll {reference_w.shape})")
    print(f"db: {'OK' if b_ok else 'FEHLER'} (Ist {actual_b.shape}, Soll {reference_b.shape})")
    if w_shape_ok:
        print("max |dW-Differenz|:", np.max(np.abs(actual_w - reference_w)))
    if b_shape_ok:
        print("max |db-Differenz|:", np.max(np.abs(actual_b - reference_b)))
    if not w_ok:
        print("dW Ist:\n", actual_w)
        print("dW Soll:\n", reference_w)
    if not b_ok:
        print("db Ist:\n", actual_b)
        print("db Soll:\n", reference_b)

if len(assignments) < len(layer_ids):
    print(f"\nHINWEIS: Nur {len(assignments)} von {len(layer_ids)} Dense-Layern geprüft.")
print("\nERGEBNIS:", "ALLE GEPRÜFTEN GRADIENTEN KORREKT" if all_ok else "ABWEICHUNGEN GEFUNDEN")
