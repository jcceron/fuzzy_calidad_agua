# Fuzzy Calidad de Agua

Proyecto de clasificación difusa y neuro-difusa (ANFIS) para la evaluación automática de la calidad del agua en sistemas de acuicultura.

## Objetivo General

Desarrollar e implementar un **sistema neuro-difuso (ANFIS)** que, a partir de parámetros fisicoquímicos y biológicos del agua, clasifique automáticamente la calidad del agua en estanques de acuicultura en tres niveles: *Excelente*, *Buena* y *Pobre*. El desempeño del sistema se validará mediante métricas difusas (*fuzzy accuracy*, *FPI*) y métricas clásicas (*accuracy*, *RMSE*), y se comparará contra un modelo de referencia (Random Forest).

## Estructura del Proyecto

```plaintext
FUZZY_CALIDAD_AGUA/
├── data/
│   ├── raw/              # Dataset original (Aquaculture – Water Quality)
│   ├── validacion/       # Conjuntos de validación y pruebas adicionales
│   └── processed/        # Conjuntos de datos procesados y winsorizados para consumo
├── env/                  # Entorno virtual Python (venv)
│   ├── Include
│   ├── Lib
│   ├── Scripts
│   └── pyvenv.cfg
├── models/
│   ├── anfis_tk.keras        # Modelo anfis keras
│   └── scaler.pkl            # Datos escalados
├── notebooks/
│   ├── models/                        # Modelos generados en el proceso
│   ├── 01_exploracion.ipynb           # Análisis exploratorio y visualización inicial
│   ├── 02_prueba_fuzzy_clasico.ipynb  # Modelo de fuzzy clásico
│   └── 03_comparativa ANSI y RF.ipynb  # Prototipado y entrenamiento ANFIS y RF
├── reports/
│   └── figuras/                  # Gráficos de MFs, curvas de entrenamiento y resultados
├── src/
│   ├── baseline/                # Modelo Random Forest de base
│   ├── pertenencia/             # Definición de funciones de pertenencia (MFs)
│   ├── reglas/                  # Generación de base de reglas difusas
│   ├── simulacion/              # Configuración de ControlSystem y simulación
│   ├── neurofuzzy/              # Implementación ANFIS (model.py, train.py, predict.py)
│   ├── utils/                   # Preprocesamiento de datos
│   └── metricas/                # Cálculo de fuzzy accuracy, FPI, RMSE y comparación con baseline
├── .gitignore                    # Ignorar env/, __pycache__/, etc.
├── requirements.txt             # Dependencias Python
└── README.md                     # Documentación del proyecto
```

## Instalación

1. Clonar el repositorio:

   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd FUZZY_CALIDAD_AGUA
   ```

2. Crear y activar el entorno virtual:

   ```bash
   python -m venv env
   # Windows
   env\Scripts\activate
   # Linux/Mac
   source env/bin/activate
   ```

3. Actualizar herramientas de instalación:

   ```bash
   python -m pip install --upgrade pip setuptools wheel
   ```

4. Instalar dependencias:

   ```bash
   pip install -r requirements.txt
   ```

## Uso

### Exploración de datos

1. Abrir y ejecutar `notebooks/01_exploracion.ipynb` para visualizar distribuciones, histogramas y comprobar la calidad del dataset.
2. Generar figuras de funciones de pertenencia preliminares en `reports/figuras/`.

### Entrenamiento ANFIS

1. Ejecutar `notebooks/02_neurofuzzy.ipynb` para:

   * Definir MFs iniciales usando umbrales normativos.
   * Configurar y entrenar la red ANFIS.
   * Monitorear la curva de pérdida (train/validation).

2. Alternativamente, desde consola:

   ```bash
   python src/neurofuzzy/train.py --data data/raw/water_quality.csv --epochs 100
   ```

## Dependencias

* Python 3.8+
* numpy
* pandas
* scikit-learn
* matplotlib
* scikit-fuzzy
* anfis
* tensorflow
* jupyterlab

## Referencias Bibliográficas

1. Veeramsetty, V., Arabelli, R., & Bernatin, T. (2024). **Aquaculture – Water Quality Dataset** \[Data set]. Mendeley Data. DOI: 10.17632/y78ty2g293.1
2. Vinay M. T., K. T. Veeramanju & Sreenivasa B. R. (2025). **Multivariate Analysis of Water Quality Parameters for Sustainable Prawn Farming**. *SEEJPH*, Vol. XXVI, S2. ISSN: 2197-5248
3. [Zimmermann, H.-J. (2010). *Fuzzy set theory*. Wiley Interdisciplinary Reviews: Computational Statistics, 2(3), 317–332.](https://doi.org/10.1002/wics.82)
4. Tajul Rosli Razak et al. (2024). **Python scikit-fuzzy: developing a fuzzy expert system for diabetes diagnosis.** *IAES International Journal of Artificial Intelligence*, 13(2), 1398–1407. DOI: 10.11591/ijai.v13.i2.pp1398-1407
5. ANZECC & ARMCANZ. (2000). **Australian and New Zealand Guidelines for Fresh and Marine Water Quality**.
6. U.S. Environmental Protection Agency (EPA). **Water Quality Criteria**.

## Autor

* Juan Carlos Cerón Lombana
* E-mail: juan.ceron@ustabuca.edu.co
* Estudiante MADSI
---


