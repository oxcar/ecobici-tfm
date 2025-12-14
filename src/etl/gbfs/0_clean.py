"""
Limpieza y conversión de datos GBFS de JSON a Parquet, con verificación e integridad.
"""
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
from pyprojroot import here

BASE_DIR = here()
LOG_DIR = Path(os.getenv("LOG_DIR", BASE_DIR / "log"))
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
GBFS_DIR = DATA_DIR / "ecobici/gbfs/raw"
PARQUET_DIR = DATA_DIR / "ecobici/gbfs/parquet"
ZIP_DIR = DATA_DIR / "ecobici/gbfs/zip"
STATIONS_CSV = DATA_DIR / "ecobici/stations/stations.csv"
TIMEZONE = ZoneInfo(os.getenv("TIMEZONE", "America/Mexico_City"))

PARQUET_DIR.mkdir(parents=True, exist_ok=True)
ZIP_DIR.mkdir(parents=True, exist_ok=True)

# Configuracion de logs
LOG_FILE = LOG_DIR / "gbfs_to_parquet.log"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(LOG_FILE)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
formatter.converter = lambda *args: datetime.now(tz=TIMEZONE).timetuple()
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def normalize_short_name(value: str) -> str:
    """Normalize GBFS short_name into zero-padded segments (e.g., '1-23' -> '001-023')."""
    if value is None:
        raise ValueError("short_name cannot be None")

    value = str(value).strip()
    if not value:
        raise ValueError("short_name cannot be empty")

    parts = value.split("-")
    padded_parts = []

    for part in parts:
        clean_part = part.strip()
        if not clean_part.isdigit():
            raise ValueError(f"Invalid short_name segment '{clean_part}' in '{value}'")
        padded_parts.append(clean_part.zfill(3))

    return "-".join(padded_parts)


def process_gbfs_to_parquet(input_root, output_root, zip_root):
    input_path = Path(input_root)
    output_path = Path(output_root)
    zip_path = Path(zip_root)

    # Leer CSV de estaciones una vez
    logger.info(f"Reading stations mapping from {STATIONS_CSV}...")
    if not STATIONS_CSV.exists():
        raise FileNotFoundError(f"Stations CSV not found: {STATIONS_CSV}")

    stations_mapping = pl.read_csv(STATIONS_CSV)
    logger.info(
        f"Loaded {stations_mapping.height} station mappings (station_id <-> station_code)"
    )

    logger.info(f"Scanning for station_status.json files in {input_path}...")
    files = sorted(list(input_path.rglob("station_status.json")))

    if not files:
        logger.info("No station_status.json files found.")
        return

    logger.info(f"Found {len(files)} files.")

    # Agrupar por dia (YYYYMMDD)
    files_by_day = {}
    for f in files:
        # Formato esperado de carpeta padre: YYYYMMDD_HHmm
        parent_name = f.parent.name
        try:
            # Extraer parte YYYYMMDD
            day_key = parent_name.split("_")[0]
            if len(day_key) != 8 or not day_key.isdigit():
                continue

            if day_key not in files_by_day:
                files_by_day[day_key] = []
            files_by_day[day_key].append(f)
        except Exception:
            continue

    logger.info(f"Found {len(files_by_day)} unique days to process.")

    today_str = datetime.now(TIMEZONE).strftime("%Y%m%d")

    for day_key, day_files in sorted(files_by_day.items()):
        if day_key == today_str:
            logger.info(f"Skipping today's data: {day_key}")
            continue

        year = day_key[:4]
        month = day_key[4:6]
        day = day_key[6:8]

        dest_dir = output_path / f"year={year}" / f"month={month}"
        dest_file = dest_dir / f"{year}_{month}_{day}.parquet"

        # Verificar si el archivo ya existe?
        # Tal vez queramos sobrescribir o saltarlo. Por ahora sobrescribimos ya que es un script ETL.

        logger.info(f"Processing {year}-{month}-{day} ({len(day_files)} snapshots)...")
        print(f"Processing {year}-{month}-{day} ({len(day_files)} snapshots)...")

        dfs = []
        for f in day_files:
            try:
                folder_name = f.parent.name  # YYYYMMDD_HHmm
                # Parsear datetime como hora local de Ciudad de Mexico (no UTC)
                dt = datetime.strptime(folder_name, "%Y%m%d_%H%M")
                # Establecer zona horaria a Ciudad de Mexico (tratando hora parseada como hora local)
                dt = dt.replace(tzinfo=TIMEZONE)

                # Leer JSON
                with open(f, "r") as jf:
                    data = json.load(jf)

                stations = data.get("data", {}).get("stations", [])
                if not stations:
                    continue

                # Crear DataFrame
                # Inferimos esquema desde la primera instantanea no vacia o dejamos que Polars lo maneje
                df = pl.DataFrame(stations, infer_schema_length=None)

                # Renombrar columnas a minusculas
                df.columns = [c.lower() for c in df.columns]
                df = df.with_columns(
                    pl.col("station_id").cast(pl.Int64),  # Asegurar que station_id es entero
                    pl.lit(dt).alias("snapshot_time"),
                )

                # Unir con stations_mapping para agregar station_code
                df = df.join(stations_mapping, on="station_id", how="left")

                # Asegurar que station_code es string
                df = df.with_columns(pl.col("station_code").cast(pl.String))

                # Verificar codigos de estacion faltantes
                missing_codes = df.filter(
                    pl.col("station_code").is_null() | (pl.col("station_code") == "")
                )
                if missing_codes.height > 0:
                    error_msg = f"Found {missing_codes.height} stations without station_code in {f}"
                    logger.error(error_msg)
                    logger.error(
                        f"Missing station IDs: {missing_codes['station_id'].unique().to_list()}"
                    )
                    raise RuntimeError(error_msg)

                dfs.append(df)

            except Exception as e:
                logger.error(f"Error processing file {f}: {e}")

        if not dfs:
            logger.info(f"No valid data for {day_key}")
            continue

        # Concatenar
        try:
            daily_df = pl.concat(dfs, how="diagonal_relaxed")

            # Verificar que el mapeo station_id <-> station_code es consistente
            mapping_check = (
                daily_df.select(["station_id", "station_code"])
                .unique()
                .group_by("station_id")
                .agg(pl.col("station_code").n_unique().alias("code_count"))
            )
            inconsistent = mapping_check.filter(pl.col("code_count") > 1)
            if inconsistent.height > 0:
                error_msg = f"Inconsistent station_id <-> station_code mapping detected for {inconsistent.height} station(s) on {day_key}"
                logger.error(error_msg)
                logger.error(
                    f"Problematic stations: {inconsistent['station_id'].to_list()}"
                )
                raise RuntimeError(error_msg)

            dest_dir.mkdir(parents=True, exist_ok=True)
            daily_df.write_parquet(dest_file)
            logger.info(f"  -> Saved to {dest_file}")

            # Verificar integridad de datos antes de eliminar origen
            # Es altamente recomendado verificar que los datos se escribieron correctamente
            written_df = pl.read_parquet(dest_file)
            if written_df.height != daily_df.height:
                raise ValueError(
                    f"Verification failed: Row count mismatch (Memory: {daily_df.height}, Disk: {written_df.height})"
                )
            logger.info(f"  -> Verification passed: {written_df.height} rows.")

            # Comprimir carpeta del dia
            # Necesitamos encontrar la carpeta del dia.
            # Basado en el primer archivo en day_files, podemos inferir la carpeta del dia.
            # Estructura: .../YYYY/MM/DD/YYYYMMDD_HHmm/station_status.json
            # day_files[0].parent es YYYYMMDD_HHmm
            # day_files[0].parent.parent es DD (la carpeta del dia)

            # Vimos antes la estructura: data/raw/gbfs/2025/11/25/20251125_0019/station_status.json
            # Entonces carpeta del dia es data/raw/gbfs/2025/11/25

            first_file = day_files[0]
            # Asumiendo estructura estandar, verificar si parent.parent coincide con el dia
            day_folder = first_file.parent.parent

            # Verificar si esta carpeta realmente corresponde al dia que acabamos de procesar
            # Podemos verificar si el nombre de carpeta coincide con 'day' (DD) o si el path termina con YYYY/MM/DD
            # Pero es mas simple confiar en la estructura si encontramos archivos ahi.

            zip_dest_dir = zip_path / f"year={year}" / f"month={month}"
            zip_dest_dir.mkdir(parents=True, exist_ok=True)
            zip_file_base = zip_dest_dir / f"{year}{month}{day}"

            logger.info(f"  -> Zipping {day_folder} to {zip_file_base}.zip")
            shutil.make_archive(str(zip_file_base), "zip", str(day_folder))

            # Eliminar carpeta original del dia
            logger.info(f"  -> Deleting original folder {day_folder}")
            shutil.rmtree(str(day_folder))

        except Exception as e:
            logger.error(f"Error concatenating/writing/zipping data for {day_key}: {e}")


if __name__ == "__main__":
    logger.info(f"\n-------------- Starting GBFS to Parquet ETL --------------")

    logger.info(f"Input Dir: {GBFS_DIR}")
    logger.info(f"Output Dir: {PARQUET_DIR}")
    logger.info(f"Zip Dir: {ZIP_DIR}")

    process_gbfs_to_parquet(GBFS_DIR, PARQUET_DIR, ZIP_DIR)
