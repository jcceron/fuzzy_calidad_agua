# src/pertenencia/thresholds.py

"""
Vértices iniciales (a, b, c, d) de las funciones de membresía
para cada variable continua.  Se usan 3 MFs:
  • 0 = 'Pobre'        (fuera del rango óptimo)
  • 1 = 'Bueno'        (en el rango normativo)
  • 2 = 'Excelente'    (centro del rango normativo)
b-c coinciden en las MF triangulares (“Bueno” y “Excelente”).

NOTA ▸ Los valores <--- *son una primera aproximación*; ajústalos
a tu experiencia y vuelve a entrenar si quieres afinar la curva.
"""

import numpy as np
# helper para triángulos
def tri(a, b, c):              # devuelve [a,b,c] -> [a,b,b,c]
    return [a, b, b, c]

MF_VERTICES = {
    #  pH ───────── 6.5-9.0 (óptimo ≈ 7.75)
    "pH": np.array([
        tri( 4.0,  5.0, 6.5),        # Pobre (bajo)
        [6.5, 6.8, 9.0, 9.3],        # Bueno (trapecio ancho)
        tri( 7.3, 7.75, 8.3),        # Excelente (pico al centro)
    ], dtype="float32"),

    #  Temperatura ───────── 25-32 °C (óptimo ≈ 28.5 °C)
    "Temperatura": np.array([
        tri(15, 22, 25),             # Pobre (fría)
        [25, 26, 32, 34],            # Bueno
        tri(27, 28.5, 30),           # Excelente
    ], dtype="float32"),

    #  Oxígeno disuelto (DO) ───────── ≥ 5 mg/L
    "DO": np.array([
        tri( 0, 2.5, 5),             # Pobre (bajo)
        [5,  5.5, 12, 15],           # Bueno
        tri( 7,  9,   11),           # Excelente (alto pero no exceso)
    ], dtype="float32"),

    #  Amoníaco no ionizado (UIA) ──── < 0.05 mg/L
    "UIA": np.array([
        tri(0.05, 0.08, 0.5),        # Pobre (alto)
        [0.00, 0.00, 0.05, 0.06],    # Bueno (trapecio muy estrecho)
        tri(0.00, 0.01, 0.02),       # Excelente
    ], dtype="float32"),

    #  Nitrato (NO3-N) ────────────── < 75 mg/L
    "Nitrato": np.array([
        tri(75, 100, 150),           # Pobre (alto)
        [0, 0, 75, 90],              # Bueno
        tri(10, 25, 40),             # Excelente
    ], dtype="float32"),

    #  Turbidez ───────────────────── ≤ 25 NTU
    "Turbidez": np.array([
        tri(25, 35, 60),             # Pobre
        [0, 0, 25, 30],              # Bueno
        tri( 0, 5, 12),              # Excelente
    ], dtype="float32"),

    #  Alcalinidad (CaCO₃) ────────── 60-150 mg/L (mín 20 mg/L)
    "Alcalinidad": np.array([
        tri(  0,  20,  60),          # Pobre (baja)
        [ 60,  70, 150, 180],        # Bueno
        tri( 80, 100, 120),          # Excelente (zona media)
    ], dtype="float32"),
}
