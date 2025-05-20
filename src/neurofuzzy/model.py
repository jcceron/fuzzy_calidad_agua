import pandas as pd
from skfuzzy.control import ControlSystem, ControlSystemSimulation

# Importa tus funciones tal como están definidas
from src.pertenencia.membership_functions import generate_universes, create_mfs
from src.reglas.rules_builder          import build_antecedents, build_consequent, build_anfis_rules

def create_fuzzy_control_system(df: pd.DataFrame, feature_cols: list):
    """
    Construye y devuelve un ControlSystemSimulation listo para inferir.
    Uso correcto de build_antecedents, build_consequent y build_anfis_rules.
    """
    # 1. Generar universos y funciones de pertenencia
    universes = generate_universes(df, feature_cols)
    mfs       = create_mfs(universes)

    # 2. Crear objetos Antecedent y Consequent
    antecedents = build_antecedents(universes, mfs)
    consequent  = build_consequent()

    # 3. Generar la lista de reglas a partir de ellos
    rules = build_anfis_rules(antecedents, consequent)

    # 4. Crear el sistema de control y la simulación
    system = ControlSystem(rules)
    sim    = ControlSystemSimulation(system)
    return sim

#______________________
# src/neurofuzzy/model.py
import tensorflow as tf
from tensorflow.keras import layers, Model, initializers

class FuzzLayer(layers.Layer):
    """
    Fuzzificación de un input continuo a n_mfs funciones de membresía
    triangulares / trapecio (parámetros entrenables).
    """
    def __init__(self, n_mfs, mf_name="trap", **kwargs):
        super().__init__(**kwargs)
        self.n_mfs = n_mfs
        self.mf_name = mf_name

    def build(self, input_shape):
        # Cada MF definida por 4 vértices (trapecio). Shape: (n_mfs, 4)
        self.p = self.add_weight(
            name="vertices",
            shape=(self.n_mfs, 4),
            initializer=initializers.RandomUniform(-1., 1.),
            trainable=True,
        )

    def call(self, x):          # x → (batch, 1)
        x = tf.expand_dims(x, -1)  # (batch, 1, 1)
        a, b, c, d = tf.unstack(self.p, axis=-1)   # (n_mfs,)
        left   = (x - a) / (b - a + 1e-6)
        mid    = 1.0
        right  = (d - x) / (d - c + 1e-6)
        mu     = tf.maximum(tf.minimum(tf.minimum(left, mid), right), 0.0)
        return mu                    # (batch, n_mfs)

def create_anfis_model(n_inputs=13, n_mfs=3, n_classes=3):
    """
    Devuelve un ANFIS 1º‑orden con salida (batch, n_classes).
    Añadimos layers.Flatten() para eliminar el eje extra.
    """
    # 1. Entradas
    inputs = [layers.Input(shape=(1,), name=f"x{i}") for i in range(n_inputs)]

    # 2. Fuzzificación
    fuzz_outputs = [FuzzLayer(n_mfs, name=f"fuzz{i}")(inp) for i, inp in enumerate(inputs)]
    fuzz_concat  = layers.Concatenate(name="concat_fuzz")(fuzz_outputs)

    # 3. Reglas + normalización (simplificada)
    rule_strength = layers.Dense(32, activation="sigmoid", name="rule_layer")(fuzz_concat)
    norm_strength = layers.LayerNormalization(name="norm")(rule_strength)

    # 4. Consecuentes lineales
    outputs = layers.Dense(n_classes, activation="linear", name="consequent")(norm_strength)

    # 5. Defuzzificación → softmax
    outputs = layers.Softmax(name="softmax")(outputs)   # (batch, 1, n_classes)
    outputs = layers.Flatten(name="flatten")(outputs)   # (batch, n_classes) ✅

    return Model(inputs=inputs, outputs=outputs, name="ANFIS")