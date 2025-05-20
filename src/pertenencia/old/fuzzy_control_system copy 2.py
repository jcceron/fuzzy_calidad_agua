# src/pertenencia/fuzzy_control_system.py
"""
Descripción:
  Implementación de un sistema difuso Mamdani para evaluar la calidad del agua
  en acuicultura, calibrado con umbrales normativos FAO/Boyd y validado con métricas
  crisp y difusas (Fuzzy Accuracy, FPI y RMSE).
Componentes:
  - Definición dinámica de Antecedentes (low/medium/high) con umbrales específicos.
  - Construcción de un Consecuente con MFs igualmente calibradas.
  - Reglas con pesos diferenciados, sin fallback predominante.
  - Inferencia eficiente reutilizando la instancia del simulador.
  - Evaluación en lote con persistencia de métricas.

Requiere:
  - numpy
  - pandas
  - scikit-fuzzy>=0.4.2
Author: <Tu Nombre>
Fecha: 2025-05-18
"""

import json
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from tqdm import tqdm

# Definición de umbrales normativos basados en FAO y Boyd
THRESHOLDS: Dict[str, Dict[str, Tuple[float, float, float]]] = {
    'Temp':                 {'low': (-np.inf, 20, 24),  'medium': (22, 26, 30),  'high': (28, 30, np.inf)},
    'Turbidity (cm)':       {'low': (50,  40, 30),     'medium': (20, 30, 50),  'high': (0,   5,  20)},
    'DO(mg/L)':             {'low': (0,    2,  4),     'medium': (3,  5,  7),   'high': (6,   8,   10)},
    'BOD (mg/L)':           {'low': (10,   8,  6),     'medium': (4,  6,  8),   'high': (0,   2,   4)},
    'CO2':                  {'low': (0,    3,  6),     'medium': (3,  6,  8),   'high': (6,   8,  np.inf)},
    'PH':                   {'low': (4.5,  5.5,6.5),   'medium': (6,  7.5,9),   'high': (8.5,10,  12)},
    'Alkalinity (mg L-1 )': {'low': (0,    40, 60),    'medium': (40,75,120),   'high': (100,150,300)},
    'Hardness (mg L-1 )':   {'low': (0,    60,100),    'medium': (60,100,180),  'high': (140,200,300)},
    'Calcium (mg L-1 )':    {'low': (0,    30, 60),    'medium': (30,60,110),   'high': (90, 120,250)},
    'Ammonia (mg L-1 )':    {'low': (0,  0.005,0.01),  'medium': (0.01,0.02,0.03),'high': (0.03,0.05,0.07)},
    'Nitrite (mg L-1 )':    {'low': (0,    0.1, 0.2),  'medium': (0.1,0.3,0.5), 'high': (0.4,1,  2)},
    'Phosphorus (mg L-1 )': {'low': (0,  0.05,0.1),    'medium': (0.1,0.3,0.5),'high': (0.4,1,  2)},
    'H2S (mg L-1 )':        {'low': (0,    0.5, 1),    'medium': (0.5,2,   5),  'high': (3,   8,   15)},
}


def build_antecedents() -> Dict[str, ctrl.Antecedent]:
    """
    Genera Antecedentes con MFs low/medium/high, reemplazando infinitos y ordenando abc.
    """
    ants: Dict[str, ctrl.Antecedent] = {}
    for var, params in THRESHOLDS.items():
        finite_vals = [v for trio in params.values() for v in trio if np.isfinite(v)]
        univ_min, univ_max = min(finite_vals), max(finite_vals)
        universe = np.linspace(univ_min, univ_max, 100)
        ant = ctrl.Antecedent(universe, var)
        for label in ['low', 'medium', 'high']:
            a, b, c = params[label]
            a = univ_min if not np.isfinite(a) else a
            c = univ_max if not np.isfinite(c) else c
            a, b, c = sorted((a, b, c))
            ant[label] = fuzz.trimf(universe, [a, b, c])
        ants[var] = ant
    return ants


def build_consequent() -> ctrl.Consequent:
    """
    Genera Consecuente 'Water Quality' con MFs poor, good y excellent.
    """
    universe = np.linspace(0, 2, 100)
    cons = ctrl.Consequent(universe, 'Water Quality')
    cons['poor']      = fuzz.trimf(universe, [0,   0,   1])
    cons['good']      = fuzz.trimf(universe, [0.8, 1.0, 1.2])
    cons['excellent'] = fuzz.trimf(universe, [1,   2,   2])
    return cons


def build_rules(ants: Dict[str, ctrl.Antecedent],
                cons: ctrl.Consequent) -> list:
    """
    Define reglas con pesos diferenciados sin parámetro weight en constructor.
    """
    rules = []
    # Regla 'Poor' 1
    r1 = ctrl.Rule(ants['DO(mg/L)']['low'] | ants['Ammonia (mg L-1 )']['high'], cons['poor'])
    r1.weight = 1.0
    rules.append(r1)
    # Regla 'Poor' 2
    r2 = ctrl.Rule(ants['Temp']['low'] | ants['H2S (mg L-1 )']['high'], cons['poor'])
    r2.weight = 0.8
    rules.append(r2)
    # Regla 'Poor' 3
    r3 = ctrl.Rule(ants['BOD (mg/L)']['high'] | ants['Turbidity (cm)']['high'], cons['poor'])
    r3.weight = 0.9
    rules.append(r3)
    # Regla 'Good'
    r4 = ctrl.Rule(ants['DO(mg/L)']['medium'] & ants['Temp']['medium'] & ~ants['Ammonia (mg L-1 )']['high'], cons['good'])
    r4.weight = 0.6
    rules.append(r4)
    # Regla 'Excellent'
    r5 = ctrl.Rule(ants['DO(mg/L)']['high'] & ants['Temp']['high'] & ~ants['Ammonia (mg L-1 )']['high'], cons['excellent'])
    r5.weight = 1.0
    rules.append(r5)
    return rules


def create_control_system(df: pd.DataFrame) -> Tuple[ctrl.ControlSystemSimulation, Dict[str, Any]]:
    """
    Construye sistema difuso y retorna simulador y metadatos.
    """
    faltantes = set(THRESHOLDS.keys()) - set(df.columns)
    assert not faltantes, f"Faltan columnas: {faltantes}"
    ants = build_antecedents()
    cons = build_consequent()
    rules = build_rules(ants, cons)
    system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system)
    return sim, {'antecedents': ants, 'consequent': cons, 'rules': rules}


def infer(sim: ctrl.ControlSystemSimulation,
          sample: Dict[str, float]) -> Tuple[float, np.ndarray]:
    """
    Ejecuta inferencia sobre una muestra y retorna (valor_crisp, vector_pertenencias).
    """
    sim.reset()
    for var, val in sample.items():
        sim.input[var] = val
    sim.compute()
    crisp = sim.output['Water Quality']
    fis = sim._fis
    membs = np.array([fis.output['Water Quality'][0][lab] for lab in ['poor', 'good', 'excellent']])
    return crisp, membs


def evaluate_dataset(df: pd.DataFrame,
                     sim: ctrl.ControlSystemSimulation,
                     show_progress: bool=False) -> Dict[str, float]:
    """
    Evalúa dataset completo, calcula métricas crisp y difusas y persiste resultados.
    """
    y_true = df['label'].values
    X = df[list(THRESHOLDS.keys())].to_dict(orient='records')
    crisps, membs = [], []
    iterator = tqdm(X, desc='Inferencia') if show_progress else X
    for sample in iterator:
        crisp, m = infer(sim, sample)
        crisps.append(crisp)
        membs.append(m)
    crisps = np.array(crisps)
    membs = np.vstack(membs)
    acc = float(np.mean(np.round(crisps) == y_true))
    fa = float(np.mean([membs[i, y_true[i]] for i in range(len(y_true))]))
    one_hot = np.eye(3)[y_true]
    fpi = float(1 - np.mean(np.linalg.norm(membs - one_hot, axis=1)))
    vals = np.array([0,1,2])
    rmse = float(np.sqrt(np.mean((crisps - vals[y_true])**2)))
    results = {'accuracy': acc, 'Fuzzy Accuracy': fa, 'FPI': fpi, 'RMSE': rmse}
    pd.DataFrame([results]).to_csv('results/metrics.csv', index=False)
    with open('results/params.json', 'w', encoding='utf-8') as f:
        json.dump(THRESHOLDS, f, ensure_ascii=False, indent=2)
    return results

# Alias para compatibilidad con notebooks previos
create_fuzzy_control_system = create_control_system
