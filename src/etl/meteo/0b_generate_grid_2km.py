"""
Genera una cuadrícula de 2 km basada en estaciones meteorológicas existentes.
Calcula los centroides de la cuadrícula y asigna estaciones a cada punto de la cuadrícula.
"""
import os
import csv
import math
from pathlib import Path

from pyprojroot import here

BASE_DIR = here()
LOG_DIR = Path(os.getenv("LOG_DIR", BASE_DIR / "log"))
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))


def create_grid_stations(input_file, grid_file, grid_size_m=2000):
    stations = []
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stations.append(row)
    except FileNotFoundError:
        print(f"Archivo de entrada no encontrado: {input_file}")
        return

    if not stations:
        print("No se encontraron estaciones.")
        return

    lats = [float(s["lat"]) for s in stations]
    lons = [float(s["lon"]) for s in stations]

    min_lat = min(lats)
    min_lon = min(lons)
    avg_lat = sum(lats) / len(lats)

    # Tamaño de cuadricula en grados
    # Latitud: 1 grado = 111111 metros => grid_size_m metros = grid_size_m/111111 grados
    lat_step = grid_size_m / 111111

    # Longitud: 1 grado = 111111 * cos(lat) metros => grid_size_m metros = grid_size_m / (111111 * cos(avg_lat))
    lon_step = grid_size_m / (111111 * math.cos(math.radians(avg_lat)))

    print(f"Origen de la cuadrícula: {min_lat}, {min_lon}")
    print(f"Paso de cuadrícula: {lat_step:.6f} deg lat, {lon_step:.6f} deg lon")

    centroids = {}

    for station in stations:
        lat = float(station["lat"])
        lon = float(station["lon"])

        # Calcular indice de cuadricula
        grid_x = math.floor((lon - min_lon) / lon_step)
        grid_y = math.floor((lat - min_lat) / lat_step)

        # Calcular centroide
        centroid_lon = min_lon + (grid_x + 0.5) * lon_step
        centroid_lat = min_lat + (grid_y + 0.5) * lat_step
        grid_id = f"{grid_x}_{grid_y}"

        station["grid_id"] = grid_id
        station["centroid_lat"] = centroid_lat
        station["centroid_lon"] = centroid_lon

        # Agregar estaciones por centroide
        if grid_id not in centroids:
            centroids[grid_id] = {
                "grid_id": grid_id,
                "latitude": centroid_lat,
                "longitude": centroid_lon,
                "station_ids": [],
            }
        centroids[grid_id]["station_ids"].append(station["station_id"])

    if centroids:
        centroid_list = list(centroids.values())

        # Ordenar por grid_id
        centroid_list.sort(key=lambda x: x["grid_id"])

        # Convertir lista de IDs a representacion string
        for c in centroid_list:
            c["station_ids"] = str(c["station_ids"])

        centroid_fieldnames = [
            "grid_id",
            "latitude",
            "longitude",
            "station_ids",
        ]
        grid_file.parent.mkdir(parents=True, exist_ok=True)
        with open(grid_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=centroid_fieldnames)
            writer.writeheader()
            writer.writerows(centroid_list)
        print(f"Guardada cuadrícula ({grid_size_m} m.) en {grid_file}")


if __name__ == "__main__":
    grid_size_m = 2000
    input_csv = DATA_DIR / "ecobici/stations/stations.csv"
    grid_csv = DATA_DIR / f"ecobici/stations/grid_{grid_size_m}.csv"

    create_grid_stations(input_csv, grid_csv, grid_size_m)
