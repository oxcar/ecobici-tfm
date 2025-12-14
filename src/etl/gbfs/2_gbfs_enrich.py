"""
Enriquecimiento de datos GBFS con múltiples enriquecedores.
"""
import os
import sys
from pathlib import Path
from typing import List

import polars as pl
from pyprojroot import here

# Agregar directorio de scripts al path
sys.path.insert(0, str(here() / "src/etl"))

from enricher.gbfs.gbfs_enricher import GBFSEnricher
from enricher.gbfs.time_enricher import CyclicTimeEnricher
from enricher.gbfs.weather_enricher import WeatherEnricher
from enricher.gbfs.gbfs_holiday_enricher import GBFSHolidayEnricher
from enricher.gbfs.lags_enricher import LagsEnricher
from enricher.stations.station_enricher import StationEnricher
from enricher.stations.station_activity_enricher import StationActivityEnricher

BASE_DIR = here()
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
INPUT_DIR = DATA_DIR / "ecobici/gbfs/2_train"
OUTPUT_DIR = DATA_DIR / "ecobici/gbfs/2_train"
WEATHER_DIR = DATA_DIR / "meteo/station/1km"
HOLIDAYS_FILE = DATA_DIR / "holidays/holidays.csv"
STATIONS_FILE = DATA_DIR / "ecobici/stations/1_stations_enriched.parquet"
STATIONS_ACTIVITY_FILE = DATA_DIR / "ecobici/stations/2_stations_activity_features.parquet"

input_path = Path(INPUT_DIR) / "0_gbfs_resampled.parquet"
output_path = Path(OUTPUT_DIR) / "1_gbfs_train.parquet"

def process_gbfs_enrich(
    input_dir: str | Path, output_dir: str | Path, enrichers: List[GBFSEnricher]
):
    """
    Enriquece los datos GBFS con varios enriquecedores.
    """
    
    if not input_path.exists():
        print(f"El archivo de entrada {input_path} no existe.")
        return
    
    print(f"Leyendo datos desde {input_path}...")
    df = pl.read_parquet(input_path)
    print(f"\nRenombrando 'num_bikes_available' a 'bikes'...")
    df = df.rename({"num_bikes_available": "bikes"})

    print(f"Cargadas {df.shape[0]:,} filas, {df.shape[1]} columnas")
    
    # Aplicar enriquecedores
    for enricher in enrichers:
        print(f"\nAplicando {enricher.__class__.__name__}...")
        df = enricher.enrich(df)
    
    print(f"\nForma final: {df.shape}")
    print(f"\nResumen de columnas:")
    print(df.head(10))
    
    # Crear directorio de salida
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\nEscribiendo datos enriquecidos en {output_path}...")
    df.write_parquet(output_path)
    print(f"Guardado: {output_path}")
    
    return df


if __name__ == "__main__":
    print(f"Directorio de entrada: {INPUT_DIR}")
    print(f"Directorio de salida: {OUTPUT_DIR}")
    print(f"Directorio meteorológico: {WEATHER_DIR}")
    print(f"Archivo de días festivos: {HOLIDAYS_FILE}")
    print(f"Archivo de estaciones: {STATIONS_FILE}\n")
    print(f"Archivo de actividad de estaciones: {STATIONS_ACTIVITY_FILE}\n")
    
    enrichers = []
    
    # Agregar enriquecedor de Tiempo Cíclico
    print("Agregando enriquecedor de Tiempo Cíclico")
    enrichers.append(CyclicTimeEnricher(time_column="snapshot_time"))
    
    # Agregar enriquecedor Estatico de Estaciones
    if STATIONS_FILE.exists():
        print("Agregando enriquecedor Estático de Estaciones")
        enrichers.append(StationEnricher(STATIONS_FILE))
    else:
        print(f"Advertencia: Archivo de estaciones {STATIONS_FILE} no encontrado, omitiendo enriquecimiento de estaciones")
    
    # Agregar enriquecedor de Actividad de Estaciones
    if STATIONS_ACTIVITY_FILE.exists():
        print("Agregando enriquecedor de Actividad de Estaciones")
        enrichers.append(StationActivityEnricher(STATIONS_ACTIVITY_FILE))
    else:
        print(f"Advertencia: Archivo de actividad de estaciones {STATIONS_ACTIVITY_FILE} no encontrado, omitiendo enriquecimiento de actividad")
    
    # Agregar enriquecedor Meteorologico
    if WEATHER_DIR.exists():
        print("Agregando enriquecedor Meteorológico")
        enrichers.append(WeatherEnricher(WEATHER_DIR))
    else:
        print(f"Advertencia: Directorio meteorológico {WEATHER_DIR} no encontrado, omitiendo enriquecimiento meteorológico")
    
    # Agregar enriquecedor de Dias Festivos
    if HOLIDAYS_FILE.exists():
        print("Agregando enriquecedor de Días Festivos")
        enrichers.append(GBFSHolidayEnricher(HOLIDAYS_FILE))
    else:
        print(f"Advertencia: Archivo de días festivos {HOLIDAYS_FILE} no encontrado, omitiendo enriquecimiento de días festivos")
    
    # Agregar enriquecedor de Rezagos Combinado (num_bikes_available + occupancy)
    print("Agregando enriquecedor de Rezagos Combinado")
    enrichers.append(LagsEnricher(
        immediate_lags=[1, 2],          # t-1, t-2 (10, 20 min)
        recent_lags=[6, 12],            # t-6, t-12 (60 min, 120 min)
        seasonal_lags=[138, 144]        # t-138, t-144 (23, 24 horas)
    ))
    
    process_gbfs_enrich(INPUT_DIR, OUTPUT_DIR, enrichers)
    
    print("\nEnriquecimiento GBFS completado!")