# src/utils/preprocessing.py

# ─── preprocessing.py ────────────────────────────────────────────────────────
import pandas as pd, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

# 1) LOCALIZAR la raíz del proyecto ───────────────────────────────────────────
#   Este archivo está en  …/src/utils/preprocessing.py
#   parents[0] = .../src/utils
#   parents[1] = .../src
#   parents[2] = .../fuzzy_calidad_agua   ← raíz del repo
ROOT = Path(__file__).resolve().parents[2]

# 2) RUTAS absolutas (independientes del cwd) ────────────────────────────────
DATA   = ROOT / "data"   / "processed" / "winsorized_water_quality.csv"
SCALER = ROOT / "models" / "scaler.pkl"

#  (Opcional) imprime una sola vez para depurar
print("[preprocessing]  ROOT =", ROOT)
print("[preprocessing]  DATA.exists? ", DATA.exists())

# 3) FUNCIÓN ------------------------------------------------------------------

def load_data(test_size=0.2, random_state=42):
    if not DATA.exists():
        raise FileNotFoundError(f"No se encontró el CSV en: {DATA}")
    df = pd.read_csv(DATA, sep=";")
    X = df.drop("Water Quality", axis=1).values.astype(float)
    y = df["Water Quality"].values.astype("int32")           # 0-1-2
    
    # Escalado igual para todos los modelos
    if SCALER.exists():
        scaler = joblib.load(SCALER)
    else:
        scaler = StandardScaler().fit(X)
        SCALER.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, SCALER)
    X = scaler.transform(X)
    return train_test_split(X, y, test_size=test_size,
                            stratify=y, random_state=random_state)
