# ── src/baseline/random_forest.py ─────────────────────────────────────
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from src.utils.preprocessing import ROOT                   # C:\fuzzy_calidad_agua

# Ruta por defecto: notebooks/models/rf.pkl
DEFAULT_PATH = ROOT / "notebooks" / "models" / "rf.pkl"


def train_rf(
    X_train,
    y_train,
    n_estimators: int = 300,
    random_state: int = 42,
    model_path: Path = DEFAULT_PATH,          # ← ahora es parámetro
):
    """Entrena un Random Forest y lo guarda en `model_path`."""
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)

    model_path.parent.mkdir(parents=True, exist_ok=True)  # crea la carpeta
    joblib.dump(rf, model_path)
    return rf


def load_rf(model_path: Path = DEFAULT_PATH):
    """Carga el Random Forest previamente guardado."""
    return joblib.load(model_path)
