# ============================================================
#  Proyecto: Predicción de la temperatura máxima del día siguiente
#  Ciudad: Caracas, Venezuela
#
#  Parte 2 - Modelado y evaluación (fases 4 y 5 de KDD)
#
#  Antes de correr este archivo hay que correr 1_preparar_datos.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs("resultados", exist_ok=True)

datos = pd.read_csv("data/caracas_diario.csv", parse_dates=["dia"])
print("Datos cargados:", datos.shape)

# ---------- Variables de entrada ----------
# Grupo 1: solo lo que pide el enunciado (humedad, presión y viento de hoy)
variables_basicas = ["humedad_media", "humedad_min", "humedad_max",
                     "presion_media", "presion_am", "presion_pm", "cambio_presion",
                     "viento_medio", "viento_max", "viento_ns", "viento_eo"]

# Grupo 2: lo anterior + temperatura de hoy, punto de rocío y época del año
variables_completas = variables_basicas + ["temp_max", "temp_min", "temp_media",
                                           "punto_rocio", "deficit_rocio",
                                           "epoca_sen", "epoca_cos"]

objetivo = "temp_max_manana"

# ---------- Separar entrenamiento y prueba ----------
# OJO: como son datos en el tiempo NO usamos train_test_split al azar,
# porque el modelo estaría "viendo el futuro". Entrenamos con 2018-2023
# y probamos con 2024-2025, que el modelo nunca vio.
entrenamiento = datos[datos["dia"] < "2024-01-01"]
prueba = datos[datos["dia"] >= "2024-01-01"]
print("Días de entrenamiento:", len(entrenamiento))
print("Días de prueba:", len(prueba))

y_entrenamiento = entrenamiento[objetivo]
y_prueba = prueba[objetivo]


def calcular_metricas(y_real, y_predicho):
    mae = mean_absolute_error(y_real, y_predicho)
    rmse = np.sqrt(mean_squared_error(y_real, y_predicho))
    r2 = r2_score(y_real, y_predicho)
    error_porcentaje = np.mean(np.abs((y_real - y_predicho) / y_real)) * 100
    aciertos_1grado = np.mean(np.abs(y_real - y_predicho) <= 1) * 100
    return mae, rmse, r2, error_porcentaje, aciertos_1grado


def crear_modelos():
    # La regresión lineal y la red neuronal necesitan que las variables
    # estén en la misma escala, por eso les ponemos StandardScaler antes.
    return {
        "Regresión lineal": make_pipeline(StandardScaler(), LinearRegression()),
        "Árbol de decisión": DecisionTreeRegressor(max_depth=5, min_samples_leaf=20, random_state=0),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10,
                                               min_samples_leaf=5, random_state=0),
        "Red neuronal": make_pipeline(StandardScaler(),
                                      MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu",
                                                   solver="adam", alpha=0.01, max_iter=2000,
                                                   early_stopping=True, random_state=0)),
    }


resultados = []
predicciones = {}

# ---------- Modelo de referencia ----------
# Si nuestro modelo no le gana a esto, no sirve de mucho:
# "mañana va a hacer la misma temperatura máxima que hoy"
pred_referencia = prueba["temp_max"]
mae, rmse, r2, ep, ac = calcular_metricas(y_prueba, pred_referencia)
resultados.append(["Referencia (mañana = hoy)", "-", mae, rmse, r2, ep, ac])

# ---------- Entrenar los 4 modelos con los 2 grupos de variables ----------
for nombre_grupo, variables in [("Básicas", variables_basicas), ("Completas", variables_completas)]:
    modelos = crear_modelos()
    for nombre_modelo, modelo in modelos.items():
        modelo.fit(entrenamiento[variables], y_entrenamiento)
        pred = modelo.predict(prueba[variables])
        mae, rmse, r2, ep, ac = calcular_metricas(y_prueba, pred)
        resultados.append([nombre_modelo, nombre_grupo, mae, rmse, r2, ep, ac])
        predicciones[(nombre_modelo, nombre_grupo)] = pred
        print(f"{nombre_modelo:20s} | {nombre_grupo:9s} | MAE = {mae:.3f} °C")

tabla = pd.DataFrame(resultados, columns=["Modelo", "Variables", "MAE", "RMSE", "R2",
                                          "Error %", "Aciertos ±1°C %"])
tabla = tabla.sort_values("MAE").round(3)
print("\n===== RESULTADOS EN EL CONJUNTO DE PRUEBA (2024-2025) =====")
print(tabla.to_string(index=False))
tabla.to_csv("resultados/comparacion_modelos.csv", index=False)

# El mejor modelo es el de menor MAE (sin contar la referencia)
mejor = tabla[tabla["Variables"] != "-"].iloc[0]
print(f"\nMejor modelo: {mejor['Modelo']} con variables {mejor['Variables']}, MAE = {mejor['MAE']:.2f} °C")

