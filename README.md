# Predicción de la temperatura máxima del día siguiente en Caracas

Proyecto de Machine Learning (regresión) que estima la temperatura máxima de **mañana** en Caracas
usando la humedad, la presión atmosférica, el viento y la temperatura de **hoy**.
Seguimos las 6 fases de la metodología KDD.

## Archivos
```
1_preparar_datos.py     # convierte los datos horarios a diarios, limpia y crea variables
2_entrenar_modelos.py   # entrena 4 modelos, los compara y genera las gráficas
3_predecir.py           # usa el modelo guardado para predecir
data/raw/caracas_maiquetia_horario.csv.gz   # datos originales de Meteostat (estación 80415)
data/caracas_diario.csv                     # dataset diario ya limpio (2018-2025)
resultados/             # gráficas, tablas y modelos (.pkl)
```

## Cómo correrlo
```bash
pip install -r requirements.txt
python 1_preparar_datos.py
python 2_entrenar_modelos.py
python 3_predecir.py               # modo interactivo
python 3_predecir.py 2025-03-10    # probar con un día del dataset
```

## Fuente de los datos
Meteostat, estación Caracas / Maiquetía Aeropuerto Internacional Simón Bolívar (WMO 80415, ICAO SVMI).
Descarga: https://bulk.meteostat.net/v2/hourly/80415.csv.gz
