"""
Extrae la lista de estaciones activas de los archivos station_information.json
de la última semana disponible de datos GBFS de Ecobici, y las guarda en un
archivo CSV.
"""
import csv
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from pyprojroot import here

BASE_DIR = here()
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))


def get_latest_week_station_info_files(base_dir):
    """Obtiene archivos station_information.json de la última semana disponible de datos."""
    station_files = []
    all_dates = []

    # Primera pasada: recolectar todas las fechas disponibles
    for year_path in base_dir.iterdir():
        if not year_path.is_dir() or not year_path.name.isdigit():
            continue

        for month_path in year_path.iterdir():
            if not month_path.is_dir() or not month_path.name.isdigit():
                continue

            for day_path in month_path.iterdir():
                if not day_path.is_dir() or not day_path.name.isdigit():
                    continue

                try:
                    folder_date = datetime(
                        int(year_path.name), int(month_path.name), int(day_path.name)
                    )
                    all_dates.append((folder_date, day_path))
                except (ValueError, OSError):
                    continue

    if not all_dates:
        return []

    # Ordenar por fecha y obtener la fecha mas reciente
    all_dates.sort(key=lambda x: x[0], reverse=True)
    latest_date = all_dates[0][0]
    cutoff_date = latest_date - timedelta(days=7)

    # Segunda pasada: recolectar archivos de la ultima semana relativa a la fecha mas reciente
    for folder_date, day_path in all_dates:
        if folder_date < cutoff_date:
            continue

        timestamps = sorted(
            [p for p in day_path.iterdir() if p.is_dir()],
            key=lambda x: x.name,
        )

        for timestamp_path in timestamps:
            file_path = timestamp_path / "station_information.json"
            if file_path.exists():
                station_files.append(file_path)

    return station_files


def extract_stations_to_csv(input_files, output_file):
    all_stations = {}  # Usar un dict para desduplicar por station_id (o short_name)
    station_code_mapping = {}  # Rastrear mapeo station_id -> station_code

    for input_file in input_files:
        print(f"Procesando {input_file}...")
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            stations = data.get("data", {}).get("stations", [])
            for station in stations:
                station_id = station.get("station_id")
                if not station_id:
                    continue

                if "short_name" in station:
                    station_code = station["short_name"]
                    station["station_code"] = station_code
                    del station["short_name"]

                    # Validar mapeo consistente
                    if station_id in station_code_mapping:
                        if station_code_mapping[station_id] != station_code:
                            raise ValueError(
                                f"Mapeo inconsistente station_id <-> station_code: "
                                f"station_id={station_id} tiene ambos "
                                f"'{station_code_mapping[station_id]}' y '{station_code}'"
                            )
                    else:
                        station_code_mapping[station_id] = station_code

                all_stations[station_id] = station

        except Exception as e:
            print(f"Error leyendo {input_file}: {e}")
            raise

    if not all_stations:
        print("No se encontraron estaciones en ningún archivo.")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Verificar si archivo de salida existe y renombrarlo
    if output_file.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_name = output_file.with_name(
            f"{output_file.stem}_{timestamp}{output_file.suffix}"
        )
        output_file.rename(new_name)
        print(f"Renombrado archivo existente a {new_name}")

    # Transformar claves a minusculas y renombrar lat/lon
    drop_columns = {
        "eightd_has_key_dispenser",
        "electric_bike_surcharge_waiver",
        "has_kiosk",
        "is_charging",
        "rental_methods",
    }
    transformed_stations = []
    for station in all_stations.values():
        transformed = {k.lower(): v for k, v in station.items()}
        # Renombrar lat -> latitude, lon -> longitude
        if "lat" in transformed:
            transformed["latitude"] = transformed.pop("lat")
        if "lon" in transformed:
            transformed["longitude"] = transformed.pop("lon")
        # Eliminar columnas no deseadas
        for col in drop_columns:
            transformed.pop(col, None)
        transformed_stations.append(transformed)

    # Ordenar por station_id
    transformed_stations.sort(key=lambda x: str(x.get("station_id", "")))

    # Construir encabezados ordenados: station_id, station_code primero,
    # name despues de capacity, external_id al final
    if transformed_stations:
        sample_keys = set(transformed_stations[0].keys())

        # Comenzar con station_id y station_code
        headers = ["station_id", "station_code"]

        # Obtener claves restantes ordenadas, excluyendo las especiales de posicionamiento
        special_keys = {"station_id", "station_code", "name", "capacity", "external_id"}
        middle_keys = sorted([k for k in sample_keys if k not in special_keys])

        # Insertar capacity y name en orden si existen
        if "capacity" in sample_keys:
            middle_keys.append("capacity")
        if "name" in sample_keys:
            middle_keys.append("name")

        headers.extend(middle_keys)

        # Agregar external_id al final si existe
        if "external_id" in sample_keys:
            headers.append("external_id")
    else:
        headers = []

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(transformed_stations)

    print(
        f"Extraídas exitosamente {len(all_stations)} estaciones únicas a {output_file}"
    )


if __name__ == "__main__":

    input_dir = DATA_DIR / "ecobici/gbfs/raw"
    processed_data_file = DATA_DIR / "ecobici/stations/stations.csv"

    try:
        latest_files = get_latest_week_station_info_files(input_dir)
        if latest_files:
            print(f"Encontrados {len(latest_files)} archivos de los últimos 7 días para procesar.")
            extract_stations_to_csv(latest_files, processed_data_file)
        else:
            print("No se encontraron archivos station_information.json en los últimos 7 días.")
    except Exception as e:
        print(f"Ocurrió un error: {e}")
