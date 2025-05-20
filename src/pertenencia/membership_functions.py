# src/pertenencia/membership_functions.py
import numpy as np
import pandas as pd
import skfuzzy as fuzz

# Tripletas [a, b, c] para Membership Functions (MFs) 'low', 'medium', 'high'
normative_thresholds = {
    'Temp': {
        'low':    [4.0, 15.0, 20.0],
        'medium': [15.0, 25.0, 30.0],
        'high':   [25.0, 30.0, 46.0]
    },
    'Turbidity (cm)': {
        'low':    [0.0, 15.0, 30.0],
        'medium': [15.0, 30.0, 60.0],
        'high':   [30.0, 60.0, 100.0]
    },
    'DO(mg/L)': {
        'low':    [0.0, 3.0, 5.0],
        'medium': [3.0, 5.0, 7.0],
        'high':   [5.0, 7.0, 10.0]
    },
    'BOD (mg/L)': {
        'low':    [0.0, 1.0, 2.0],
        'medium': [1.0, 2.0, 4.0],
        'high':   [2.0, 4.0, 8.0]
    },
    'CO2': {
        'low':    [0.2, 3.0, 5.0],
        'medium': [3.0, 5.0, 8.0],
        'high':   [5.0, 8.0, 13.0]
    },
    'PH': {
        'low':    [4.0, 6.5, 7.5],
        'medium': [6.5, 7.5, 8.5],
        'high':   [7.5, 8.5, 12.5]
    },
    'Alkalinity (mg L-1 )': {
        'low':    [25.0, 40.0, 75.0],
        'medium': [40.0, 75.0, 150.0],
        'high':   [75.0, 150.0, 271.0]
    },
    'Hardness (mg L-1 )': {
        'low':    [50.0, 75.0, 100.0],
        'medium': [75.0, 100.0, 200.0],
        'high':   [100.0, 200.0, 302.0]
    },
    'Calcium (mg L-1 )': {
        'low':    [20.0, 30.0, 60.0],
        'medium': [30.0, 60.0, 120.0],
        'high':   [60.0, 120.0, 253.0]
    },
    'Ammonia (mg L-1 )': {
        'low':    [0.0, 0.005, 0.012],
        'medium': [0.005, 0.012, 0.03],
        'high':   [0.012, 0.03, 0.08]
    },
    'Nitrite (mg L-1 )': {
        'low':    [0.0, 0.01, 0.1],
        'medium': [0.01, 0.1, 1.0],
        'high':   [0.1, 1.0, 2.9]
    },
    'Phosphorus (mg L-1 )': {
        'low':    [0.0, 0.05, 0.5],
        'medium': [0.05, 0.5, 2.0],
        'high':   [0.5, 2.0, 4.9]
    },
    'H2S (mg L-1 )': {
        'low':    [0.0, 0.005, 0.01],
        'medium': [0.005, 0.01, 0.02],
        'high':   [0.01, 0.02, 0.033]
    }
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
    Genera funciones de pertenencia triangulares usando los parámetros normativos.
    """
    mfs = {}
    for col, x in universes.items():
        if col in normative_thresholds:
            params = normative_thresholds[col]
            mfs[col] = {
                'low':    fuzz.trimf(x, params['low']),
                'medium': fuzz.trimf(x, params['medium']),
                'high':   fuzz.trimf(x, params['high'])
            }
        else:
            # Fallback cuartiles (en caso de nuevas variables)
            Q1, Q2, Q3 = np.percentile(x, [25, 50, 75])
            min_v, max_v = x.min(), x.max()
            mfs[col] = {
                'low':    fuzz.trimf(x, [min_v, min_v, Q1]),
                'medium': fuzz.trimf(x, [Q1, Q2, Q3]),
                'high':   fuzz.trimf(x, [Q3, max_v, max_v])
            }
    return mfs
