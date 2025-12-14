"""
Promedio espacial de datos meteorológicos de cuadrícula 1km a nivel de estación.
"""
import os
import sys
from pathlib import Path

import polars as pl
from pyprojroot import here

BASE_DIR = here()
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
STATIONS_PATH = DATA_DIR / "ecobici/stations/stations.csv"
GRID_PARQUET_DIR = DATA_DIR / "meteo/grid/1km"
OUTPUT_DIR = DATA_DIR / "meteo/station/1km_mean"


def main():
    if not STATIONS_PATH.exists():
        print(f"Error: Archivo de estaciones no encontrado en {STATIONS_PATH}")
        sys.exit(1)

    if not GRID_PARQUET_DIR.exists():
        print(f"Error: Directorio parquet de grid no encontrado en {GRID_PARQUET_DIR}")
        sys.exit(1)

    print("Cargando datos meteorológicos de grid 1km...")
    try:
        weather_lazy = pl.scan_parquet(
            str(GRID_PARQUET_DIR / "**/*.parquet"), hive_partitioning=True
        )
    except Exception as e:
        print(f"Error escaneando archivos parquet de clima: {e}")
        sys.exit(1)

    # Obtener lista de columnas de valor (excluir claves)
    schema = weather_lazy.collect_schema()
    exclude_cols = ["grid_id", "grid", "grid_point", "time", "date"]
    
    # Identificar columnas continuas para promediar
    continuous_cols = [
        col
        for col in schema.names()
        if col not in exclude_cols
    ]

    print(f"Columnas a promediar: {continuous_cols}")

    print("Calculando promedio espacial global (media de todos los puntos de grid por marca de tiempo)...")
    
    # Agrupar por tiempo y calcular media para todas las columnas continuas
    # Esto colapsa la dimension espacial, resultando en una serie temporal representando toda el area
    global_mean_df = (
        weather_lazy
        .group_by("time")
        .agg([pl.col(c).mean() for c in continuous_cols])
        .sort("time")
        .collect()
    )

    # Convertir tiempo a zona horaria America/Mexico_City
    print("Convirtiendo tiempo a zona horaria America/Mexico_City...")
    global_mean_df = global_mean_df.with_columns(
        pl.col("time")
        .str.to_datetime()
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone("America/Mexico_City")
    )

    print(f"Promedio global calculado. Pasos de tiempo: {len(global_mean_df)}")

    print("Cargando lista de estaciones...")
    stations_df = pl.read_csv(
        STATIONS_PATH, schema_overrides={"station_code": pl.String}
    ).select("station_code")
    
    unique_stations = stations_df["station_code"].to_list()
    print(f"Encontradas {len(unique_stations)} estaciones.")

    print(f"Escribiendo datos promediados a {OUTPUT_DIR}...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Iterar sobre estaciones y escribir el mismo dataframe para cada una
    for i, station_code in enumerate(unique_stations):
        if (i + 1) % 10 == 0:
            print(f"Procesando [{i+1}/{len(unique_stations)}]...", end="\r")

        # Agregar columna station_code
        st_df = global_mean_df.with_columns(
            pl.lit(str(station_code)).alias("station_code")
        )

        # Definir ruta de salida
        station_dir = OUTPUT_DIR / f"station={station_code}"
        station_dir.mkdir(parents=True, exist_ok=True)
        out_file = station_dir / "weather.parquet"

        # Escribir
        st_df.write_parquet(out_file)

    print(f"\nHecho. Generados datos meteorológicos promediados para {len(unique_stations)} estaciones.")


if __name__ == "__main__":
    main()
