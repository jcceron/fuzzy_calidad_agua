# src/pertenencia/fuzzy_control_system.py
"""
Implementación del sistema fuzzy clásico (Mamdani) para evaluación de calidad del agua.
Define Antecedentes, Consecuente y reglas compuestas basadas en multicolinealidad y correlaciones.
"""

import numpy as np
import pandas as pd
import skfuzzy as fuzz
import skfuzzy.control as ctrl
from skfuzzy.control import Antecedent, Consequent, Rule, ControlSystem, ControlSystemSimulation

from .membership_functions import generate_universes, create_mfs


def build_antecedents(universes: dict, mfs: dict) -> dict:
    """
    Crea objetos Antecedent para cada variable de entrada, asignando MFs 'low', 'medium' y 'high'.
    """
    antecedents = {}
    for var, x in universes.items():
        ant = Antecedent(x, var)
        for label in ['low', 'medium', 'high']:
            ant[label] = mfs[var][label]
        antecedents[var] = ant
    return antecedents


def build_consequent() -> Consequent:
    """
    Crea el objeto Consequent 'Water Quality' con tres MFs triangulares: 'poor', 'good', 'excellent'.
    Universo de 0 a 2 (índice de calidad).
    """
    quality_range = np.linspace(0, 2, 101)
    cons = Consequent(quality_range, 'Water Quality')
    cons['poor']      = fuzz.trimf(quality_range, [0.0, 0.0, 1.0])
    cons['good']      = fuzz.trimf(quality_range, [0.5, 1.0, 1.5])
    cons['excellent'] = fuzz.trimf(quality_range, [1.0, 2.0, 2.0])
    return cons


def build_composite_rules(antecedents: dict, consequent: Consequent) -> list:

    """
    Construye reglas compuestas basadas en análisis de correlaciones:
    - Combina DO, pH y Temp para excelencia o pobreza.
    - Agrupa Ammonia y Nitrite para detectar contaminación.
    - Incluye relaciones BOD-Nitrite y Turbidity-H2S.
    - Añade regla de fallback para garantizar salida (una variable en 'medium').
    """

    rules = []
    
    # Regla 1: DO alta AND pH medio AND Temp media -> excelente
    rules.append(Rule(
        antecedents['DO(mg/L)']['high'] & antecedents['PH']['medium'] & antecedents['Temp']['medium'],
        consequent['excellent']
    ))
    # Regla 2: DO baja OR pH baja OR Temp baja -> pobre
    rules.append(Rule(
        antecedents['DO(mg/L)']['low'] | antecedents['PH']['low'] | antecedents['Temp']['low'],
        consequent['poor']
    ))
    # Regla 3: Ammonia alta OR Nitrite alta -> pobre
    rules.append(Rule(
        antecedents['Ammonia (mg L-1 )']['high'] | antecedents['Nitrite (mg L-1 )']['high'],
        consequent['poor']
    ))
    # Regla 4: BOD alta AND Nitrite alta -> pobre
    rules.append(Rule(
        antecedents['BOD (mg/L)']['high'] & antecedents['Nitrite (mg L-1 )']['high'],
        consequent['poor']
    ))
    # Regla 5: Turbidity alta AND H2S baja -> good
    rules.append(Rule(
        antecedents['Turbidity (cm)']['high'] & antecedents['H2S (mg L-1 )']['low'],
        consequent['good']
    ))
    # Regla 6: CO2 baja AND BOD baja -> good
    rules.append(Rule(
        antecedents['CO2']['low'] & antecedents['BOD (mg/L)']['low'],
        consequent['good']
    ))
    # Regla 7: DO alta AND pH alta AND Temp alta -> excelente
    rules.append(Rule(
        antecedents['DO(mg/L)']['high'] & antecedents['PH']['high'] & antecedents['Temp']['high'],
        consequent['excellent']
    ))
    # Regla 8: Fallback - si cualquier variable en 'medium' -> good
    #medium_combination = None
    #for ant in antecedents.values():
    #    if medium_combination is None:
    #        medium_combination = ant['medium']
    #    else:
    #        medium_combination = medium_combination | ant['medium']
    #rules.append(Rule(
    #    medium_combination,
    #    consequent['good']
    #))

    return rules


def create_fuzzy_control_system(df: pd.DataFrame, feature_cols: list) -> ControlSystemSimulation:
    """
    Construye y devuelve un simulador del sistema de control fuzzy (Mamdani) con reglas compuestas.
    """
    universes = generate_universes(df, feature_cols, n_points=200)
    mfs = create_mfs(universes)
    antecedents = build_antecedents(universes, mfs)
    consequent = build_consequent()
    rules = build_composite_rules(antecedents, consequent)

    system = ControlSystem(rules)
    sim = ControlSystemSimulation(system)
    return sim


def infer_fuzzy(sim: ControlSystemSimulation, x: list, feature_cols: list) -> float:
    """
    Ejecuta la inferencia difusa para un vector de entrada x.
    Crea una nueva simulación para cada llamada.
    Retorna el valor defuzzificado de la salida..
    """
    
    ctrl_system = getattr(sim, 'ctrl', None)
    if ctrl_system is None:
        raise RuntimeError("Simulador inválido: no contiene 'ctrl'.")
    new_sim = ControlSystemSimulation(ctrl_system)
    for idx, var in enumerate(feature_cols):
        try:
            new_sim.input[var] = float(x[idx])
        except (KeyError, ValueError):
            continue
    new_sim.compute()
    output_key = next(iter(new_sim.output.keys()), None)
    if output_key is None:
        raise RuntimeError("No se generó salida tras la inferencia.")
    return new_sim.output[output_key]
