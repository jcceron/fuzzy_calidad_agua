# src/pertenencia/membership_functions.py
import numpy as np
import pandas as pd
import skfuzzy as fuzz

# Diccionario de umbrales normativos (min, bajo, alto, max)
normative_thresholds = {
    'PH': [6.5, 6.5, 9.0, 9.0],
    'Temp': [0.0, 25.0, 32.0, 32.0],                    # °C 
    'DO(mg/L)': [0.0, 5.0, 10.0, 10.0],                 # mg/L
    'Ammonia (mg L-1 )': [0.0, 0.0, 0.05, 0.05],        # mg NH₃–N/L
    'Nitrite (mg L-1 )': [0.0, 0.0, 0.1, 0.1],          # mg NO₃–N/L
    'Turbidity (cm)': [0.0, 0.0, 25.0, 25.0],           # NTU
    'Alkalinity (mg L-1 )': [20.0, 60.0, 150.0, 150.0]  # mg CaCO₃/L
}

def generate_universes(df: pd.DataFrame, feature_cols: list, n_points: int = 100) -> dict:
    """
    Crea universos de discurso para cada variable basada en su rango en el DataFrame.
    """
    universes = {}
    for col in feature_cols:
        min_v = df[col].min()
        max_v = df[col].max()
        universes[col] = np.linspace(min_v, max_v, n_points)
    return universes


def create_mfs(universes: dict) -> dict:
    """
    Genera funciones de pertenencia triangulares/trapezoidales usando umbrales normativos.
    Para variables sin umbrales normativos, usa cuartiles.
    """
    mfs = {}
    for col, x in universes.items():
        if col in normative_thresholds:
            mn, low, high, mx = normative_thresholds[col]
            mfs[col] = {
                'low':    fuzz.trimf(x, [mn, mn, low]),
                'medium': fuzz.trapmf(x, [mn, low, high, mx]),
                'high':   fuzz.trimf(x, [high, mx, mx])
            }
        else:
            # Fallback cuartiles
            Q1 = float(pd.Series(x).quantile(0.25))
            Q2 = float(pd.Series(x).quantile(0.50))
            Q3 = float(pd.Series(x).quantile(0.75))
            min_v = float(x.min())
            max_v = float(x.max())
            mfs[col] = {
                'low':    fuzz.trimf(x, [min_v, min_v, Q1]),
                'medium': fuzz.trimf(x, [Q1, Q2, Q3]),
                'high':   fuzz.trimf(x, [Q3, max_v, max_v])
            }
    return mfs
