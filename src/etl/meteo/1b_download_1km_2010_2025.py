"""
Descarga datos meteorológicos horarios de Open-Meteo para una cuadrícula de 1km entre 2010 y 2025.
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
SAVE_DIR = DATA_DIR / "meteo/grid/1km_all_years"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
TIMEZONE = os.getenv("TIMEZONE", "America/Mexico_City")

grid_csv_path = DATA_DIR / "meteo/grid/grid_1km.csv"
if not grid_csv_path.exists():
    raise FileNotFoundError(f"Archivo de grid no encontrado en {grid_csv_path}")

grid_df = pl.read_csv(grid_csv_path)
GRID_POINTS = [
    {"id": str(row["grid_id"]), "lat": row["latitude"], "lon": row["longitude"]}
    for row in grid_df.iter_rows(named=True)
]

START_DATE = "2010-01-01"
END_DATE = datetime.now(ZoneInfo(TIMEZONE)).strftime("%Y-%m-%d")

# Variables seleccionadas
VARIABLES = [
    "temperature_2m",
    "apparent_temperature",
    "rain",
    "snowfall",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "relative_humidity_2m",
]


def build_url(point):
    base = "https://archive-api.open-meteo.com/v1/archive?"

    params = (
        f"latitude={point['lat']}"
        f"&longitude={point['lon']}"
        f"&start_date={START_DATE}"
        f"&end_date={END_DATE}"
        f"&hourly={','.join(VARIABLES)}"
        f"&timezone={TIMEZONE}"
    )
    return base + params


def download_period(point, retries=3):
    url = build_url(point)

    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=60)

            if r.status_code != 200:
                try:
                    error_msg = r.json().get("reason", r.text)
                except Exception:
                    error_msg = r.text
                print(
                    f"[ERROR {point['id']}] Estado {r.status_code} - {error_msg}"
                )
                continue

            data = r.json()

            if "hourly" not in data:
                print(f"[VACÍO {point['id']}] Sin datos")
                return None

            df = pl.DataFrame(data["hourly"])
            df = df.with_columns([
                pl.lit(point["id"]).alias("grid_point"),
                pl.col("time").str.to_datetime().dt.replace_time_zone(TIMEZONE, non_existent='null', ambiguous='earliest').alias("time")
            ])

            return df

        except Exception as e:
            print(f"[EXCEPCIÓN {point['id']}] Intento {attempt}/{retries}: {e}")
            time.sleep(2)

    return None


def save_parquet(point_id, df):
    # Crear directorio de particion: grid=grid_id
    partition_dir = SAVE_DIR / f"grid={point_id}"
    partition_dir.mkdir(parents=True, exist_ok=True)

    filename = partition_dir / "meteo.parquet"
    df.write_parquet(filename)
    print(f"[OK] Guardado {filename}")


def download_all():
    # Calcular filas esperadas desde START_DATE hasta END_DATE
    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    end = datetime.strptime(END_DATE, "%Y-%m-%d")
    days = (end - start).days + 1
    expected_rows = days * 24

    for point in GRID_POINTS:
        print(f"\nDescargando punto {point['id']} (lat={point['lat']}, lon={point['lon']})")

        # Verificar si el archivo ya existe
        partition_dir = SAVE_DIR / f"grid={point['id']}"
        filename = partition_dir / "meteo.parquet"

        if filename.exists():
            try:
                existing_df = pl.read_parquet(filename)
                if existing_df.height == expected_rows:
                    print(f"[SALTAR] Archivo ya existe y está completo ({existing_df.height} filas)")
                    continue
                else:
                    print(
                        f"[RE-DESCARGAR] Archivo existe pero está incompleto ({existing_df.height}/{expected_rows} filas)"
                    )
            except Exception as e:
                print(f"[RE-DESCARGAR] Error leyendo {filename}: {e}")

        print(f"  Descargando periodo {START_DATE} a {END_DATE}...")
        df = download_period(point)

        if df is not None:
            if df.height != expected_rows:
                print(
                    f"[INCOMPLETO] {point['id']}: {df.height}/{expected_rows} filas. Guardando de todas formas."
                )

            save_parquet(point["id"], df)
        else:
            print(f"[FALLO] No se pudo descargar {point['id']}")

        time.sleep(1)  # Evitar saturar la API

    print("\nDescarga completada")


if __name__ == "__main__":
    # Calcular el END_DATE actual
    end_date_display = datetime.now(ZoneInfo(TIMEZONE)).strftime("%Y-%m-%d")
    
    print("="*70)
    print("DESCARGA DE DATOS METEOROLÓGICOS - HISTÓRICO COMPLETO")
    print("="*70)
    print(f"Periodo: {START_DATE} a {end_date_display}")
    print(f"Variables: {', '.join(VARIABLES)}")
    print(f"Puntos de grid: {len(GRID_POINTS)}")
    print(f"Destino: {SAVE_DIR}")
    print("="*70)
    print()
    
    download_all()
