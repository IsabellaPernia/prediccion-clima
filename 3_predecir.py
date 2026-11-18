# ============================================================
#  Proyecto: Predicción de la temperatura máxima del día siguiente
#  Ciudad: Caracas, Venezuela
#
#  Parte 3 - Usar el modelo (fase 6 de KDD)
#
#  Uso:  python 3_predecir.py 2025-03-10
#        Toma un día del dataset, predice con el modelo completo y
#        lo compara con lo que pasó en realidad.
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



if __name__ == "__main__":
    if len(sys.argv) > 1:
        probar_con_fecha(sys.argv[1])
    else:
        print("Uso: python 3_predecir.py AAAA-MM-DD")
