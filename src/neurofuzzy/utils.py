# src/neurofuzzy/utils.py
import tensorflow as tf
from src.pertenencia.thresholds import MF_VERTICES
from src.neurofuzzy.model import FuzzLayer   # importa tu clase exacta

def initialise_mfs(model):
    """
    Recorre las capas FuzzLayer y pone los vértices según el diccionario.
    El nombre de la capa debe ser "fuzz<Var>" o similar.
    """
    for layer in model.layers:
        if isinstance(layer, FuzzLayer):
            # • Si la nombraste 'fuzz0', 'fuzz1'… crea tu propio mapa
            #   var_name = VARIABLES[int(layer.name.lstrip("fuzz"))]
            var_name = layer.name.split("_")[-1]   # ej. 'fuzz_pH' -> 'pH'
            if var_name in MF_VERTICES:
                layer.p.assign(MF_VERTICES[var_name])
