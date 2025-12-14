"""
Descarga datos meteorológicos horarios de Open-Meteo para una cuadrícula de 2km entre 2010 y 2025.
Guarda los datos en formato Parquet particionado por punto de cuadrícula y año.
"""
import calendar
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import requests
from pyprojroot import here

BASE_DIR = here()
LOG_DIR = Path(os.getenv("LOG_DIR", BASE_DIR / "log"))
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
SAVE_DIR = DATA_DIR / "meteo/grid/parquet"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
TIMEZONE = os.getenv("TIMEZONE", "America/Mexico_City")

grid_csv_path = DATA_DIR / "meteo" / "grid_2km.csv"
if not grid_csv_path.exists():
    raise FileNotFoundError(f"Archivo de grid no encontrado en {grid_csv_path}")

grid_df = pl.read_csv(grid_csv_path)
GRID_POINTS = [
    {"id": str(row["grid_id"]), "lat": row["latitude"], "lon": row["longitude"]}
    for row in grid_df.iter_rows(named=True)
]

START_YEAR = 2010
END_YEAR = 2025

VARIABLES = [
    "temperature_2m",
    "apparent_temperature",
    "weather_code",
    "rain",
    "snowfall",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "relative_humidity_2m",
    "dew_point_2m",
]


def build_url(point, year):
    base = "https://archive-api.open-meteo.com/v1/archive?"

    if year == 2025:
        end_date = "2025-11-01"
    else:
        end_date = f"{year}-12-31"

    params = (
        f"latitude={point['lat']}"
        f"&longitude={point['lon']}"
        f"&start_date={year}-01-01"
        f"&end_date={end_date}"
        f"&hourly={','.join(VARIABLES)}"
        f"&timezone={TIMEZONE}"
    )
    return base + params


def download_year(point, year, retries=3):
    url = build_url(point, year)

    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=60)

            if r.status_code != 200:
                try:
                    error_msg = r.json().get("reason", r.text)
                except Exception:
                    error_msg = r.text
                print(
                    f"[ERROR {point['id']}] Año {year}: Estado {r.status_code} - {error_msg}"
                )
                continue

            data = r.json()

            if "hourly" not in data:
                print(f"[VACÍO {point['id']}] Año {year} sin datos")
                return None

            df = pl.DataFrame(data["hourly"])
            df = df.with_columns(pl.lit(point["id"]).alias("grid_point"))

            return df

        except Exception as e:
            print(f"[EXCEPCIÓN {point['id']}] Intento {attempt}/{retries}: {e}")
            time.sleep(2)

    return None


def save_parquet(point_id, year, df):
    # Crear directorio de particion: grid=grid_id
    partition_dir = SAVE_DIR / f"grid={point_id}"
    partition_dir.mkdir(parents=True, exist_ok=True)

    filename = partition_dir / f"{year}.parquet"
    df.write_parquet(filename)
    print(f"[OK] Guardado {filename}")


def download_all():
    for point in GRID_POINTS:
        print(f" Descargando punto {point['id']}")

        for year in range(START_YEAR, END_YEAR + 1):
            # Verificar si el archivo ya existe
            partition_dir = SAVE_DIR / f"grid={point['id']}"
            filename = partition_dir / f"{year}.parquet"

            if filename.exists():
                try:
                    existing_df = pl.read_parquet(filename)
                    if year == 2025:
                        expected_rows = 7320  # Hasta 1 de Noviembre
                    else:
                        expected_rows = 8784 if calendar.isleap(year) else 8760

                    if existing_df.height == expected_rows:
                        print(
                            f"[SALTAR] Año {year} ya existe y está completo en {filename}"
                        )
                        continue
                    else:
                        print(
                            f"[RE-DESCARGAR] Año {year} existe pero está incompleto ({existing_df.height}/{expected_rows})."
                        )
                except Exception as e:
                    print(f"[RE-DESCARGAR] Error leyendo {filename}: {e}")

            print(f"> Año {year}...")
            df = download_year(point, year)

            if df is not None:
                # Verificar completitud (datos horarios para todo el año)
                if year == 2025:
                    expected_rows = 7320  # Hasta 1 Nov
                else:
                    expected_rows = 8784 if calendar.isleap(year) else 8760

                if df.height != expected_rows:
                    print(
                        f"[INCOMPLETO] {point['id']} año {year}: {df.height}/{expected_rows} filas. No se guarda."
                    )
                    continue

                save_parquet(point["id"], year, df)
            else:
                print(f"[FALLO] No se pudo descargar {point['id']} año {year}")

            time.sleep(1)  # Evita saturar la API


if __name__ == "__main__":
    download_all()
