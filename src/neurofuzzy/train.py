import tensorflow as tf, joblib, argparse
from src.utils.preprocessing import load_data
from src.neurofuzzy.utils import initialise_mfs
from src.metricas.metrics import fuzzy_accuracy_tf, fpi

from pathlib import Path
from sklearn.metrics import confusion_matrix
import numpy as np, pandas as pd

OUT = Path("../../notebooks/models/anfis_tf")


# Esta función para el sistema fuzzy estático.

def train_fuzzy_system(sim, X: np.ndarray, y: np.ndarray, feature_cols: list):
    """
    Para un sistema fuzzy estático no hay entrenamiento de parámetros,
    pero podemos usar esta función para evaluar performance sobre un
    conjunto de entrenamiento (p. ej. calcular fuzzy accuracy).
    
    Parámetros:
      - sim: ControlSystemSimulation
      - X:   matriz de inputs
      - y:   vector de etiquetas reales (0,1,2)
      - feature_cols: lista de nombres de columnas
    
    Retorna:
      - y_pred: vector de salidas crisp
    """
    y_pred = []
    for xi in X:
        y_pred.append(infer_fuzzy(sim, xi, feature_cols))
    return y_pred

# Estas funciones para el entrenamiento híbrido

from src.neurofuzzy.model import create_anfis_model

def fuzzy_accuracy(y_true, y_pred):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true,1),
                                           tf.argmax(y_pred,1)), tf.float32))

def train(args):
    # 1️⃣  datos
    X_tr, X_ts, y_tr_int, y_ts_int = load_data()   # ← enteros 0/1/2

    # one-hot para loss & métricas
    y_tr = tf.one_hot(y_tr_int, 3)
    y_ts = tf.one_hot(y_ts_int, 3)

    # 2️⃣  modelo
    anfis = create_anfis_model(
        n_inputs=X_tr.shape[1],
        n_mfs=args.mfs,
        n_classes=3
    )
    try:
        initialise_mfs(anfis)
    except NameError:
        pass

    # 3️⃣  compile  (usa categorical_crossentropy)
    anfis.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss="categorical_crossentropy",
        metrics=[fuzzy_accuracy_tf, fpi]
    )

    # 4️⃣  entrenamiento
    history = anfis.fit(
        [X_tr[:, i] for i in range(X_tr.shape[1])],
        y_tr,
        validation_data=(
            [X_ts[:, i] for i in range(X_ts.shape[1])],
            y_ts
        ),
        epochs=args.epochs,
        batch_size=32,
        verbose=2
    )

    # 5️⃣  guardar en formato .keras
    out_path = Path("models/anfis_tf.keras")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anfis.save(out_path)
    print(f"Modelo guardado en {out_path}")

    # 6️⃣  matriz de confusión
    y_pred = np.argmax(
        anfis.predict([X_ts[:, i] for i in range(X_ts.shape[1])]),
        axis=1
    )
    cm = confusion_matrix(tf.argmax(y_ts,1), y_pred)
    pd.DataFrame(
        cm,
        index=["Pobre","Bueno","Excelente"],
        columns=["Pred_P","Pred_B","Pred_E"]
    ).to_csv("reports/cm_anfis.csv")
    print("Confusion matrix guardada en reports/cm_anfis.csv")


# ── CLI launcher (igual) ─────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr",     type=float, default=1e-3)
    p.add_argument("--mfs",    type=int,   default=3)
    train(p.parse_args())