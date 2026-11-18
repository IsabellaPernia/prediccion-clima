# ============================================================
#  Proyecto: Predicción de la temperatura máxima del día siguiente
#  Ciudad: Caracas, Venezuela
#
#  Parte 3 - Usar el modelo (fase 6 de KDD)
#
#  Opción 1:  python 3_predecir.py
#             Pide la humedad, presión y viento de hoy y predice la máxima de mañana.
#  Opción 2:  python 3_predecir.py 2025-03-10
#             Toma un día del dataset, predice con el modelo completo y
#             lo compara con lo que pasó en realidad.
# ============================================================

import sys
import joblib
import numpy as np
import pandas as pd


def probar_con_fecha(fecha):
    modelo, variables = joblib.load("resultados/modelo_completo.pkl")
    datos = pd.read_csv("data/caracas_diario.csv", parse_dates=["dia"])
    fila = datos[datos["dia"] == fecha]
    if len(fila) == 0:
        print("No hay datos para esa fecha. Pruebe con una entre 2018-07-20 y 2025-12-30.")
        return
    prediccion = modelo.predict(fila[variables])[0]
    real = fila["temp_max_manana"].values[0]
    print(f"Día {fecha}: máxima de hoy {fila['temp_max'].values[0]} °C, "
          f"humedad media {fila['humedad_media'].values[0]:.0f} %, "
          f"presión media {fila['presion_media'].values[0]:.1f} hPa, "
          f"viento medio {fila['viento_medio'].values[0]:.1f} km/h")
    print(f"Predicción para mañana: {prediccion:.1f} °C")
    print(f"Lo que pasó en realidad: {real:.1f} °C  (error de {abs(prediccion - real):.1f} °C)")


def modo_interactivo():
    modelo, variables = joblib.load("resultados/modelo_basico.pkl")
    print("Ingrese los datos de HOY:")
    humedad_media = float(input("  Humedad media (%): "))
    humedad_min = float(input("  Humedad mínima (%): "))
    humedad_max = float(input("  Humedad máxima (%): "))
    presion_am = float(input("  Presión en la mañana (hPa): "))
    presion_pm = float(input("  Presión en la tarde (hPa): "))
    viento_medio = float(input("  Velocidad media del viento (km/h): "))
    viento_max = float(input("  Velocidad máxima del viento (km/h): "))
    direccion = float(input("  Dirección del viento en grados (0 = norte, 90 = este): "))

    # Las mismas transformaciones que en 1_preparar_datos.py
    entrada = pd.DataFrame([{
        "humedad_media": humedad_media,
        "humedad_min": humedad_min,
        "humedad_max": humedad_max,
        "presion_media": (presion_am + presion_pm) / 2,
        "presion_am": presion_am,
        "presion_pm": presion_pm,
        "cambio_presion": presion_pm - presion_am,
        "viento_medio": viento_medio,
        "viento_max": viento_max,
        "viento_ns": viento_medio * np.cos(np.radians(direccion)),
        "viento_eo": viento_medio * np.sin(np.radians(direccion)),
    }])
    prediccion = modelo.predict(entrada[variables])[0]
    print(f"\nTemperatura máxima estimada para mañana: {prediccion:.1f} °C")
    print("(Ojo: este modelo usa solo humedad, presión y viento, así que es poco preciso.")
    print(" Para mejores resultados use la opción con fecha, que usa todas las variables.)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        probar_con_fecha(sys.argv[1])
    else:
        modo_interactivo()
