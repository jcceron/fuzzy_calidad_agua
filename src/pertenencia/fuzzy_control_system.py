# src/pertenencia/fuzzy_control_system.py
"""
Descripción:
  Sistema difuso Mamdani para evaluar la calidad del agua en acuicultura,
  calibrado con umbrales normativos importados de thresholds.py y validado con métricas crisp y difusas.

Componentes:
  - Importar `normative_thresholds` con 13 variables: Temp, Turbidity (cm), DO(mg/L),
    BOD (mg/L), CO2, PH, Alkalinity (mg L-1 ), Hardness (mg L-1 ), Calcium (mg L-1 ),
    Ammonia (mg L-1 ), Nitrite (mg L-1 ), Phosphorus (mg L-1 ), H2S (mg L-1 ).
  - Generar Antecedentes dinámicamente con MFs low/medium/high (trimf).
  - Construir Consecuente 'Water Quality' con MFs poor, good, excellent.
  - Definir reglas con inclusión de CO2, PH y pesos diferenciados.
  - Inferencia eficiente reutilizando el simulador.
  - Evaluación en lote con cálculo y persistencia de métricas (accuracy, Fuzzy Accuracy, FPI, RMSE).

Requisitos:
  - numpy
  - pandas
  - scikit-fuzzy>=0.4.2
  - thresholds.py en el path, definiendo `normative_thresholds`
  - tqdm (opcional para barra de progreso)

Autor: <Tu Nombre>
Fecha: 2025-05-18
"""
import json
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from tqdm import tqdm

# Importar umbrales normativos definidos en thresholds.py
try:
    from src.pertenencia.thresholds import normative_thresholds as THRESHOLDS
except ImportError:
    raise ImportError("No se encontró thresholds.py con 'normative_thresholds'.")


def build_antecedents() -> Dict[str, ctrl.Antecedent]:
    """
    Genera Antecedentes con MFs low/medium/high para cada variable en THRESHOLDS.
    Reemplaza infinitos por los extremos del universo.
    """
    ants: Dict[str, ctrl.Antecedent] = {}
    for var, params in THRESHOLDS.items():
        # Extraer valores finitos para determinar universe
        vals = [v for trio in params.values() for v in trio if np.isfinite(v)]
        umin, umax = min(vals), max(vals)
        universe = np.linspace(umin, umax, 100)
        ant = ctrl.Antecedent(universe, var)
        for label in ['low', 'medium', 'high']:
            a, b, c = params[label]
            a = umin if not np.isfinite(a) else a
            c = umax if not np.isfinite(c) else c
            a, b, c = sorted((a, b, c))
            ant[label] = fuzz.trimf(universe, [a, b, c])
        ants[var] = ant
    return ants


def build_consequent() -> ctrl.Consequent:
    """
    Genera Consecuente 'Water Quality' con tres MFs triangular.
    """
    universe = np.linspace(0, 2, 100)
    cons = ctrl.Consequent(universe, 'Water Quality')
    cons['poor']      = fuzz.trimf(universe, [0,   0,   1])
    cons['good']      = fuzz.trimf(universe, [0.8, 1.0, 1.2])
    cons['excellent'] = fuzz.trimf(universe, [1,   2,   2])
    return cons


def build_rules(ants: Dict[str, ctrl.Antecedent], cons: ctrl.Consequent) -> list:
    """
    Define reglas con pesos diferenciados e incluye cada antecedente para su registro.
    """
    rules = []
    # Regla 'Poor' por CO2 alto
    r0 = ctrl.Rule(
        ants['CO2']['high'], 
        cons['poor']); r0.weight=0.7; rules.append(r0)
    # Reglas 'Poor' principales
    r1 = ctrl.Rule(
        ants['DO(mg/L)']['low'] | 
        ants['Ammonia (mg L-1 )']['high'], 
        cons['poor']); r1.weight=1.0; rules.append(r1)
    r2 = ctrl.Rule(
        ants['Temp']['low'] | 
        ants['H2S (mg L-1 )']['high'], 
        cons['poor']); r2.weight=0.8; rules.append(r2)
    r3 = ctrl.Rule(
        ants['BOD (mg/L)']['high'] | 
        ants['Turbidity (cm)']['high'], 
        cons['poor']); r3.weight=0.9; rules.append(r3)
    # Regla 'Good'
    r4 = ctrl.Rule(
        ants['DO(mg/L)']['medium'] & 
        ants['Temp']['medium'] & 
        ants['Ammonia (mg L-1 )']['high'], 
        cons['good']); r4.weight=0.6; rules.append(r4)
    # Regla 'Excellent'
    r5 = ctrl.Rule(
        ants['DO(mg/L)']['high'] & 
        ants['Temp']['high'], 
        cons['excellent']); r5.weight=1.0; rules.append(r5)
    # Dummy rules para registrar todos los antecedentes
    for var, ant in ants.items():
        # Cada antecedente se referencia al menos una vez con una regla de bajo peso
        rd = ctrl.Rule(
            ant['medium'], 
            cons['good']); rd.weight = 0.01
        rules.append(rd)
    return rules


def create_control_system(df: pd.DataFrame) -> Tuple[ctrl.ControlSystemSimulation, Dict[str, Any]]:
    """
    Construye sistema difuso Mamdani y retorna simulador y metadatos.

    Parámetros:
      df: DataFrame que debe contener todas las columnas listadas en THRESHOLDS.

    Retorna:
      sim: ControlSystemSimulation para inferencia.
      meta: dict con claves 'antecedents', 'consequent', 'rules'.
    """
    falt = set(THRESHOLDS.keys()) - set(df.columns)
    assert not falt, f"Faltan columnas: {falt}"
    ants = build_antecedents()
    cons = build_consequent()
    rules = build_rules(ants, cons)
    system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system)
    return sim, {'antecedents': ants, 'consequent': cons, 'rules': rules}


def infer(sim: ctrl.ControlSystemSimulation,
          sample: Dict[str, float],
          cons: ctrl.Consequent) -> Tuple[float, np.ndarray]:
    """
    Realiza inferencia y calcula grados de pertenencia usando las MFs del Consecuente.
    """
    sim.reset()
    for var, val in sample.items():
        sim.input[var] = val
    sim.compute()
    crisp = sim.output['Water Quality']

    # Calcular pertenencias usando las MFs definidas en el Consecuente
    universe = cons.universe
    memberships = []
    for label in ['poor', 'good', 'excellent']:
        mf = cons.terms[label].mf
        memberships.append(fuzz.interp_membership(universe, mf, crisp))
    return crisp, np.array(memberships)

def evaluate_dataset(df: pd.DataFrame,
                     sim: ctrl.ControlSystemSimulation,
                     meta: Dict[str, Any],
                     label_col: str = 'Water Quality',
                     show_progress: bool = False) -> Dict[str, float]:
    """
    Evalúa el dataset completo, calcula métricas crisp y difusas y persiste resultados.

    Parámetros:
      df: DataFrame con variables y columna de etiquetas.
      sim: simulador de sistema difuso.
      label_col: nombre de columna con etiquetas numéricas {0,1,2}.
      show_progress: True para mostrar barra tqdm.

    Retorna:
      Diccionario {'accuracy', 'Fuzzy Accuracy', 'FPI', 'RMSE'}.
    """
    assert label_col in df.columns, f"Columna '{label_col}' no existe"
    y_true = df[label_col].values
    X = df[list(THRESHOLDS)].to_dict('records')
    crisps, membs = [], []
    it = tqdm(X, desc='Inferencia') if show_progress else X
    for s in it:
        c, m = infer(sim, s, meta['consequent'])  # PASA meta['consequent'] a infer
        crisps.append(c)
        membs.append(m)
    crisps = np.array(crisps)
    membs = np.vstack(membs)
    acc = float(np.mean(np.round(crisps) == y_true))
    fa = float(np.mean([membs[i, y_true[i]] for i in range(len(y_true))]))
    oh = np.eye(3)[y_true]
    fpi = float(1 - np.mean(np.linalg.norm(membs - oh, axis=1)))
    rmse = float(np.sqrt(np.mean((crisps - np.array([0, 1, 2])[y_true]) ** 2)))
    res = {'accuracy': acc, 'Fuzzy Accuracy': fa, 'FPI': fpi, 'RMSE': rmse}
    pd.DataFrame([res]).to_csv('../reports/metrics.csv', index=False)
    with open('../reports/params.json', 'w', encoding='utf-8') as f:
        json.dump(THRESHOLDS, f, indent=2)
    return res

# Alias para compatibilidad con notebooks previos
create_fuzzy_control_system = create_control_system
