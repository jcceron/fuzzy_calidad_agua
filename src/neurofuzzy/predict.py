# src/neurofuzzy/predict.py

from skfuzzy.control import ControlSystemSimulation

def infer_fuzzy(sim: ControlSystemSimulation,
                x: list,
                feature_cols: list) -> float:
    """
    Ejecuta la inferencia difusa para un vector de entrada x.
    Asigna solamente las variables que estén definidas como Antecedent
    en el simulador, y omite las demás sin error.

    Parámetros:
      - sim: instancia de ControlSystemSimulation ya configurada
      - x:   lista o arreglo 1D con valores de entrada
      - feature_cols: lista de nombres de variables en el mismo orden que x

    Retorna:
      - valor defuzzificado de la variable de salida (float)
    """
    # 1. Asignar cada input si existe
    for idx, col in enumerate(feature_cols):
        try:
            sim.input[col] = float(x[idx])
        except (ValueError, KeyError):
            # La variable no está en el sistema fuzzy: la ignoramos
            continue

    # 2. Ejecutar el motor de inferencia
    sim.compute()

    # 3. Tomar la primera (y única) variable de salida
    output_var = list(sim.output.keys())[0]
    return sim.output[output_var]
