# src/reglas/rules_builder.py

import numpy as np
import skfuzzy as fuzz
from skfuzzy.control import Antecedent, Consequent, Rule

def build_antecedents(universes: dict, mfs: dict) -> dict:
    """
    Construye un diccionario de objetos Antecedent, cada uno con tres
    funciones de pertenencia ('low', 'medium', 'high').

    Parámetros:
    -----------
    universes : dict
        Mapea cada nombre de variable a su universo de discurso (array de puntos).
    mfs : dict
        Mapea cada nombre de variable a sus MFs:
        {'low': array, 'medium': array, 'high': array}.

    Retorna:
    --------
    antecedents : dict
        Mapea cada nombre de variable al objeto Antecedent correspondiente.
    """
    antecedents = {}
    for col, uni in universes.items():
        ant = Antecedent(uni, col)
        ant['bajo']     = mfs[col]['low']
        ant['medio']    = mfs[col]['medium']
        ant['alto']     = mfs[col]['high']
        antecedents[col] = ant
    return antecedents


def build_consequent() -> Consequent:
    """
    Construye el objeto Consequent para la salida 'Calidad de Agua', con tres
    conjuntos difusos etiquetados como 'Pobre', 'Bueno' y 'Excelente'.

    El universo de salida es [0, 1, 2], donde:
      0 → Pobre
      1 → Bueno
      2 → Excelente

    Retorna:
    --------
    consequent : Consequent
        Objeto Consequent con tres MFs definidas.
    """
    # Universo desde 0 hasta 2, paso 1
    y_universe = np.arange(0, 3, 1)
    cons = Consequent(y_universe, 'Calidad del Agua')

    # Definición de MFs:
    # Pobre      : se activa en torno a 0
    # Bueno      : se activa en torno a 1
    # Excelente  : se activa en torno a 2

    cons['Pobre']      = fuzz.trimf(cons.universe, [0, 0, 1])
    cons['Bueno']      = fuzz.trimf(cons.universe, [0, 1, 2])
    cons['Excelente'] = fuzz.trimf(cons.universe, [1, 2, 2])

    return cons


def build_anfis_rules(antecedents: dict, consequent: Consequent) -> list:
    """
    Genera la lista de reglas difusas que relacionan combinaciones de
    condiciones en las variables de entrada con la calidad del agua.

    Las reglas se basan en umbrales normativos previamente definidos
    en las MFs de los antecedents.

    Parámetros:
    -----------
    antecedents : dict
        Diccionario de objetos Antecedent (con MFs 'low', 'medium', 'high').
    consequent : Consequent
        Objeto Consequent para 'Water Quality'.

    Retorna:
    --------
    rules : list[Rule]
        Lista de objetos Rule configurados.
    """
    rules = []

    # R0: bajo por CO2 alto
    rules.append(
        Rule(
            antecedents['CO2']['alto'],
            consequent['Pobre'],
            label='R0_CO2_high'
        )
    )

    # R1: Si DO es bajo O Ammonia es alto → Poor
    rules.append(
        Rule(
            antecedents['DO(mg/L)']['bajo'] |
            antecedents['Ammonia (mg L-1 )']['alto'],
            consequent['Pobre'],
            label='R1_DO_low_or_Ammonia_high'
        )
    )

    # R2: Si pH está en rango medio Y DO está en rango medio → Good
    rules.append(
        Rule(
            antecedents['Temp']['bajo'] &
            antecedents['H2S (mg L-1 )']['alto'],
            consequent['Pobre'],
            label='R2_Temp_bajo_and_H2S_alto'
        )
    )

    # R3: Si pH es alto Y Turbidity es bajo → Excellent
    rules.append(
        Rule(
            antecedents['BOD (mg/L)']['alto'] &
            antecedents['Turbidity (cm)']['alto'],
            consequent['Pobre'],
            label='R3_BOD_high_and_Turbidity_alto'
        )
    )

    # R4: Si Alkalinity está en rango medio Y Temp en rango medio → Good
    rules.append(
        Rule(
            antecedents['DO(mg/L)']['medio'] &
            antecedents['Temp']['medio'] &
            antecedents["Ammonia (mg L-1 )"]['alto'],
            consequent['Bueno'],
            label='R4_DO_medium_and_Temp_medium_Amonia_alto'
        )
    )

    # R5: Si Nitrite es alto O Turbidity es alto → Poor
    rules.append(
        Rule(
            antecedents['DO(mg/L)']['alto'] &
            antecedents['Temp']['alto'],
            consequent['Excelente'],
            label='R5_DO_high_Temp_high'
        )
    )

    # Aquí podrías añadir más reglas basadas en tu conocimiento acuícola,
    # por ejemplo combinaciones con pH, Ammonia, Hardness, etc.

    return rules
