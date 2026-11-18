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

# ---------- 3. Crear nuevas variables ----------
# Cambio de presión durante el día (si baja, puede venir mal tiempo)
diario["cambio_presion"] = diario["presion_pm"] - diario["presion_am"]
# Diferencia entre temperatura y punto de rocío (aire seco = diferencia grande)
diario["deficit_rocio"] = diario["temp_media"] - diario["punto_rocio"]
# Época del año. Usamos seno y coseno para que diciembre quede "cerca" de enero
dia_del_anio = diario["dia"].dt.dayofyear
diario["epoca_sen"] = np.sin(2 * np.pi * dia_del_anio / 365)
diario["epoca_cos"] = np.cos(2 * np.pi * dia_del_anio / 365)

# ---------- 4. Variable objetivo: temperatura máxima de MAÑANA ----------
diario = diario.sort_values("dia").reset_index(drop=True)
diario["temp_max_manana"] = diario["temp_max"].shift(-1)
# Solo es válido si la siguiente fila es de verdad el día siguiente
dias_hasta_siguiente = diario["dia"].shift(-1) - diario["dia"]
diario.loc[dias_hasta_siguiente != pd.Timedelta(days=1), "temp_max_manana"] = np.nan
diario = diario.dropna(subset=["temp_max_manana"])

# Algunos días les falta la presión de la mañana o de la tarde.
# Rellenamos con la presión media de ese mismo día.
diario["presion_am"] = diario["presion_am"].fillna(diario["presion_media"])
diario["presion_pm"] = diario["presion_pm"].fillna(diario["presion_media"])
diario["cambio_presion"] = diario["presion_pm"] - diario["presion_am"]
diario = diario.dropna()

diario = diario.drop(columns=["horas_registradas", "horas_presion"])
columnas_numericas = diario.select_dtypes("number").columns
diario[columnas_numericas] = diario[columnas_numericas].round(2)
diario.to_csv("data/caracas_diario.csv", index=False)

print("\nDataset final:", diario.shape)
print("Desde", diario["dia"].min().date(), "hasta", diario["dia"].max().date())
print(diario[["temp_max", "humedad_media", "presion_media", "viento_medio", "temp_max_manana"]].describe().round(2))
print("\nGuardado en data/caracas_diario.csv")
