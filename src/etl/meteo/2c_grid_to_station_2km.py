"""
Interpolación de datos meteorológicos de cuadrícula a estaciones usando IDW y Vecino Más Cercano.
"""
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
from pyprojroot import here
from sklearn.neighbors import BallTree

BASE_DIR = here()
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
GRID_MAPPING_PATH = DATA_DIR / "ecobici/stations/grid_2000.csv"
STATIONS_PATH = DATA_DIR / "ecobici/stations/stations.csv"
GRID_PARQUET_DIR = DATA_DIR / "meteo/grid/parquet"
STATION_PARQUET_DIR = DATA_DIR / "meteo/station/parquet"
STATION_PARQUET_DIR.mkdir(parents=True, exist_ok=True)


def calculate_idw_weights(
    stations_df: pl.DataFrame, grid_df: pl.DataFrame, k: int = 4
) -> pl.DataFrame:
    """
    Calcula los pesos de Ponderación de Distancia Inversa (IDW) para cada estación
    basado en los k puntos de cuadrícula más cercanos.
    """
    # Extraer coordenadas (lat, lon)
    # BallTree espera [lat, lon] en radianes para metrica haversine
    station_coords = np.radians(stations_df.select(["latitude", "longitude"]).to_numpy())
    grid_coords = np.radians(grid_df.select(["latitude", "longitude"]).to_numpy())

    # Construir BallTree
    tree = BallTree(grid_coords, metric="haversine")

    # Consultar k vecinos mas cercanos
    # Devuelve distancias (en radianes) e indices
    dists, indices = tree.query(station_coords, k=k)

    # Convertir distancias a metros (radio aprox de Tierra 6371km)
    dists_m = dists * 6371000

    weights_list = []

    grid_ids = grid_df["grid_id"].to_list()
    station_ids = stations_df["station_code"].to_list()

    for i, station_id in enumerate(station_ids):
        d = dists_m[i]
        idx = indices[i]

        # Manejar coincidencia exacta (distancia ~ 0)
        # Si alguna distancia es muy pequeña, asignar peso 1 a ese punto y 0 a otros
        epsilon = 1e-6
        if np.any(d < epsilon):
            w = np.where(d < epsilon, 1.0, 0.0)
        else:
            # IDW potencia 2
            w = 1.0 / (d**2)

        # Normalizar pesos
        w = w / np.sum(w)

        for j in range(k):
            weights_list.append(
                {"station_id": station_id, "grid_id": grid_ids[idx[j]], "weight": w[j]}
            )

    return pl.DataFrame(weights_list)


def main():
    if not GRID_MAPPING_PATH.exists():
        print(f"Error: Archivo de grid no encontrado en {GRID_MAPPING_PATH}")
        sys.exit(1)

    if not STATIONS_PATH.exists():
        print(f"Error: Archivo de estaciones no encontrado en {STATIONS_PATH}")
        sys.exit(1)

    print("Cargando estaciones y puntos de grid...")

    # Cargar estaciones
    stations_df = pl.read_csv(
        STATIONS_PATH, schema_overrides={"station_code": pl.String}
    ).select(["station_code", "latitude", "longitude"])

    # Cargar puntos de grid
    grid_df = pl.read_csv(GRID_MAPPING_PATH).select(
        ["grid_id", "latitude", "longitude"]
    )

    print(f"Calculando pesos IDW (k=4) para {len(stations_df)} estaciones...")
    weights_df = calculate_idw_weights(stations_df, grid_df, k=4)

    print("Cargando datos meteorológicos...")
    # Carga lazy de todos los datos meteorologicos
    # La estructura es grid_id=X/part-*.parquet, así que particionamiento Hive podría funcionar automáticamente
    # si escaneamos el directorio raíz. Pero la estructura de carpeta es `grid={grid_id}`.
    # Polars scan_parquet con hive_partitioning=True debería manejarlo.

    try:
        weather_lazy = pl.scan_parquet(
            str(GRID_PARQUET_DIR / "**/*.parquet"), hive_partitioning=True
        )
    except Exception as e:
        print(f"Error escaneando archivos parquet de clima: {e}")
        sys.exit(1)

    # Renombrar 'grid' a 'grid_id' si es necesario para coincidir con weights_df
    # La clave de particion es probablemente 'grid' basado en estructura de carpeta 'grid=...'
    if "grid" in weather_lazy.columns and "grid_id" not in weather_lazy.columns:
        weather_lazy = weather_lazy.rename({"grid": "grid_id"})

    # Obtener lista de columnas de valor (excluir claves)
    schema = weather_lazy.collect_schema()

    # Definir manejo de columnas basado en entrada del usuario
    # weather_code es categorica -> usar Vecino Mas Cercano (peso maximo)
    # grid_point, grid, grid_id, time -> excluir de interpolacion
    categorical_cols = ["weather_code"]
    exclude_cols = ["grid_id", "grid", "grid_point", "time", "date"]

    continuous_cols = [
        col
        for col in schema.names()
        if col not in categorical_cols and col not in exclude_cols
    ]

    print(f"Columnas continuas (Interpolación IDW): {continuous_cols}")
    print(f"Columnas categóricas (Vecino más cercano): {categorical_cols}")

    # Procesar interpolacion...

    # Unir datos meteorologicos con pesos
    # weather: [grid_id, time, val1, val2...]
    # weights: [station_id, grid_id, weight]
    # joined: [station_id, grid_id, time, val1, val2..., weight]

    joined_lazy = weather_lazy.join(weights_df.lazy(), on="grid_id", how="inner")

    # Necesitamos escribir un archivo por estacion.
    # Recolectar todo el dataset podría ser pesado (500 estaciones * 10 años * horario).
    # 500 * 87600 = 43 millones de filas.
    # Es mejor iterar sobre estaciones y procesar.

    unique_stations = stations_df["station_code"].to_list()

    for i, station_code in enumerate(unique_stations):
        print(
            f"[{i+1}/{len(unique_stations)}] Procesando estación {station_code}...",
            end="\r",
        )

        # Filtrar pesos para esta estacion
        st_weights = weights_df.filter(pl.col("station_id") == station_code)
        st_grid_ids = st_weights["grid_id"].to_list()

        # Filtrar datos meteorologicos para estas cuadriculas
        # Usamos los IDs de cuadricula especificos para minimizar lectura de datos
        st_weather = weather_lazy.filter(pl.col("grid_id").is_in(st_grid_ids))

        # Unir con pesos especificos
        st_joined = st_weather.join(st_weights.lazy(), on="grid_id")

        # Agregar
        # Continuas: Suma Ponderada
        agg_exprs = [
            (pl.col(c) * pl.col("weight")).sum().alias(c) for c in continuous_cols
        ]

        # Categoricas: Valor del punto de cuadricula con el peso mas alto (Vecino Mas Cercano)
        # Ordenamos por peso descendente y tomamos el primero
        agg_exprs.extend(
            [
                pl.col(c).sort_by("weight", descending=True).first().alias(c)
                for c in categorical_cols
                if c in schema.names()
            ]
        )

        st_result = st_joined.group_by("time").agg(agg_exprs).sort("time").collect()

        # Agregar columna station_code al resultado
        st_result = st_result.with_columns([
            pl.lit(station_code).alias("station_code")
        ])

        # Escribir
        station_dir = STATION_PARQUET_DIR / f"station={station_code}"
        station_dir.mkdir(parents=True, exist_ok=True)
        out_file = station_dir / "weather.parquet"
        st_result.write_parquet(out_file)

    print(f"\nHecho. Procesadas {len(unique_stations)} estaciones.")


if __name__ == "__main__":
    main()
