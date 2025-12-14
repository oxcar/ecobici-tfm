"""
Enriquecimiento de datos de viajes con información adicional como festivos.
"""
import os
import sys
from pathlib import Path
from typing import List

import polars as pl
from pyprojroot import here

# Agregar el directorio de scripts al path para poder importar enricher
BASE_DIR = here()
SCRIPTS_DIR = BASE_DIR / "src" / "etl"
sys.path.insert(0, str(SCRIPTS_DIR))

from enricher.enricher import TripEnricher
from enricher.trips.holiday_enricher import HolidayEnricher

LOG_DIR = Path(os.getenv("LOG_DIR", BASE_DIR / "log"))
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))


def process_trips(
    trips_dir: str | Path, output_dir: str | Path, enrichers: List[TripEnricher]
):
    trips_path = Path(trips_dir)
    output_path = Path(output_dir)

    if not trips_path.exists():
        print(f"El directorio de viajes {trips_path} no existe.")
        return

    # Encontrar todos los archivos parquet recursivamente
    parquet_files = sorted(list(trips_path.rglob("*.parquet")))

    if not parquet_files:
        print(f"No se encontraron archivos parquet en {trips_dir}")
        return

    print(f"Encontrados {len(parquet_files)} archivos parquet para procesar.")

    for file_path in parquet_files:
        # Calcular ruta relativa para mantener estructura
        rel_path = file_path.relative_to(trips_path)
        dest_path = output_path / rel_path

        # Crear directorios padre
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Procesando {file_path.name}...")

        try:
            # Leer archivo Parquet de viajes
            df = pl.read_parquet(file_path)

            # Aplicar todos los enriquecedores
            for enricher in enrichers:
                df = enricher.enrich(df)

            # Escribir a salida
            df.write_parquet(dest_path)
            print(f"  - Guardado en {dest_path}")

        except Exception as e:
            print(f"Error procesando {file_path}: {e}")


if __name__ == "__main__":

    trips_dir = DATA_DIR / "ecobici/trips/1_clean"
    holidays_path = DATA_DIR / "holidays/holidays.csv"
    output_dir = DATA_DIR / "ecobici/trips/2_enriched"

    print(f"Directorio de viajes: {trips_dir}")
    print(f"Archivo de festivos: {holidays_path}")
    print(f"Directorio de salida: {output_dir}")

    holiday_enricher = HolidayEnricher(holidays_path)
    process_trips(trips_dir, output_dir, [holiday_enricher])
