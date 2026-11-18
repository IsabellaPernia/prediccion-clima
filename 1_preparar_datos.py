# ============================================================
#  Proyecto: Predicción de la temperatura máxima del día siguiente
#  Ciudad: Caracas, Venezuela (estación Maiquetía, SVMI / WMO 80415)
#
#  Parte 1 - Preparación de los datos (fases 2 y 3 de KDD)
#
#  Los datos vienen de Meteostat y son HORARIOS (una fila por hora).
#  Aquí los convertimos a datos DIARIOS y creamos la columna que
#  queremos predecir: la temperatura máxima de mañana.
# ============================================================

import pandas as pd
import numpy as np

# ---------- 1. Cargar los datos horarios ----------
# El archivo de Meteostat no trae encabezado, así que ponemos los nombres
# según la documentación de Meteostat (https://dev.meteostat.net/bulk/hourly.html)
columnas = ["fecha", "hora", "temp", "punto_rocio", "humedad", "lluvia", "nieve",
            "dir_viento", "vel_viento", "rafaga", "presion", "sol", "condicion"]

datos = pd.read_csv("data/raw/caracas_maiquetia_horario.csv.gz", header=None, names=columnas)
print("Datos horarios originales:", datos.shape)

# Meteostat guarda la hora en UTC. Caracas está en UTC-4, así que
# restamos 4 horas para que cada día empiece a la medianoche local.
datos["fecha_hora"] = pd.to_datetime(datos["fecha"]) + pd.to_timedelta(datos["hora"], unit="h")
datos["fecha_hora"] = datos["fecha_hora"] - pd.Timedelta(hours=4)
datos["dia"] = datos["fecha_hora"].dt.normalize()
datos["hora_local"] = datos["fecha_hora"].dt.hour

# La presión casi no se registraba antes de 2018, por eso usamos 2018-2025
datos = datos[(datos["dia"] >= "2018-02-01") & (datos["dia"] <= "2025-12-31")]
print("Datos horarios 2018-2025:", datos.shape)

# Revisamos cuántos nulos hay
print("\nPorcentaje de nulos por columna:")
print((datos.isnull().mean() * 100).round(1))

# La lluvia, nieve, ráfaga, sol y condición tienen demasiados nulos -> se descartan
datos = datos.drop(columns=["lluvia", "nieve", "rafaga", "sol", "condicion"])

# ---------- 2. Pasar de datos horarios a diarios ----------
# El viento tiene dirección (grados), y no se puede sacar el promedio
# directo (el promedio de 350° y 10° no es 180°). Por eso lo separamos
# en dos componentes: norte-sur y este-oeste.
rad = np.radians(datos["dir_viento"])
datos["viento_ns"] = datos["vel_viento"] * np.cos(rad)
datos["viento_eo"] = datos["vel_viento"] * np.sin(rad)

# Presión de la mañana (6 a 11 h) y de la tarde (12 a 17 h)
datos["presion_am"] = datos["presion"].where(datos["hora_local"].between(6, 11))
datos["presion_pm"] = datos["presion"].where(datos["hora_local"].between(12, 17))

diario = datos.groupby("dia").agg(
    temp_max=("temp", "max"),
    temp_min=("temp", "min"),
    temp_media=("temp", "mean"),
    humedad_media=("humedad", "mean"),
    humedad_min=("humedad", "min"),
    humedad_max=("humedad", "max"),
    punto_rocio=("punto_rocio", "mean"),
    presion_media=("presion", "mean"),
    presion_am=("presion_am", "mean"),
    presion_pm=("presion_pm", "mean"),
    viento_medio=("vel_viento", "mean"),
    viento_max=("vel_viento", "max"),
    viento_ns=("viento_ns", "mean"),
    viento_eo=("viento_eo", "mean"),
    horas_registradas=("temp", "count"),
    horas_presion=("presion", "count"),
).reset_index()

# Si un día tiene pocas horas medidas, la máxima puede salir mal
# (por ejemplo si faltan las horas del mediodía). Pedimos al menos 18 horas.
print("\nDías totales:", len(diario))
diario = diario[(diario["horas_registradas"] >= 18) & (diario["horas_presion"] >= 12)]
print("Días con suficientes mediciones:", len(diario))

