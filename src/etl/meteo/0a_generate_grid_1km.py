"""
Genera una cuadrícula de 1 km que rodea todos los puntos en el archivo de estaciones de entrada.
"""
import os
import csv
import math
from pathlib import Path

from pyprojroot import here

BASE_DIR = here()
LOG_DIR = Path(os.getenv("LOG_DIR", BASE_DIR / "log"))
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))


def create_1km_grid(input_file, output_file):
    """
    Genera una cuadrícula de 1km que rodea todos los puntos en el archivo de estaciones de entrada.
    
    Args:
        input_file: Ruta a stations.csv con columnas: station_id, latitude, longitude
        output_file: Ruta para guardar el CSV de la cuadrícula generada
    """
    # Leer estaciones
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

    # Extraer coordenadas
    lats = [float(s["latitude"]) for s in stations]
    lons = [float(s["longitude"]) for s in stations]

    min_lat = min(lats)
    max_lat = max(lats)
    min_lon = min(lons)
    max_lon = max(lons)
    
    avg_lat = (min_lat + max_lat) / 2

    print(f"Límites de estaciones:")
    print(f"  Latitud:  [{min_lat:.6f}, {max_lat:.6f}]")
    print(f"  Longitud: [{min_lon:.6f}, {max_lon:.6f}]")

    # Tamaño de cuadricula: 1km = 1000 metros
    grid_size_m = 1000

    # Calcular pasos en grados para 1km
    # Latitud: 1 grado ≈ 111,111 metros
    lat_step = grid_size_m / 111111

    # Longitud: 1 grado ≈ 111,111 * cos(latitud) metros
    lon_step = grid_size_m / (111111 * math.cos(math.radians(avg_lat)))

    print(f"\nParámetros de cuadrícula:")
    print(f"  Tamaño de cuadrícula: {grid_size_m} metros")
    print(f"  Paso de latitud:  {lat_step:.6f} grados")
    print(f"  Paso de longitud: {lon_step:.6f} grados")

    # Calcular limites de cuadricula (extender ligeramente mas alla de limites de estaciones)
    grid_min_lat = math.floor(min_lat / lat_step) * lat_step
    grid_max_lat = math.ceil(max_lat / lat_step) * lat_step
    grid_min_lon = math.floor(min_lon / lon_step) * lon_step
    grid_max_lon = math.ceil(max_lon / lon_step) * lon_step

    print(f"\nLímites de cuadrícula:")
    print(f"  Latitud:  [{grid_min_lat:.6f}, {grid_max_lat:.6f}]")
    print(f"  Longitud: [{grid_min_lon:.6f}, {grid_max_lon:.6f}]")

    # Generar celdas de cuadricula
    grid_cells = []
    grid_id = 0

    lat = grid_min_lat
    y_index = 0
    while lat < grid_max_lat:
        lon = grid_min_lon
        x_index = 0
        while lon < grid_max_lon:
            # Calcular centroide de celda
            centroid_lat = lat + lat_step / 2
            centroid_lon = lon + lon_step / 2

            grid_cells.append({
                "grid_id": grid_id,
                "grid_x": x_index,
                "grid_y": y_index,
                "latitude": centroid_lat,
                "longitude": centroid_lon,
                "min_latitude": lat,
                "max_latitude": lat + lat_step,
                "min_longitude": lon,
                "max_longitude": lon + lon_step,
            })

            grid_id += 1
            x_index += 1
            lon += lon_step
        
        y_index += 1
        lat += lat_step

    print(f"\nGeneradas {len(grid_cells)} celdas de cuadrícula")

    # Calcular cobertura aproximada
    lat_range = grid_max_lat - grid_min_lat
    lon_range = grid_max_lon - grid_min_lon
    
    # Convertir a km
    lat_range_km = lat_range * 111.111
    lon_range_km = lon_range * 111.111 * math.cos(math.radians(avg_lat))
    
    print(f"Cobertura de cuadrícula: {lat_range_km:.2f} km × {lon_range_km:.2f} km")

    # Guardar a CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        "grid_id",
        "grid_x",
        "grid_y",
        "latitude",
        "longitude",
        "min_latitude",
        "max_latitude",
        "min_longitude",
        "max_longitude",
    ]
    
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(grid_cells)
    
    print(f"\nCuadricula guardada en: {output_file}")


if __name__ == "__main__":
    input_csv = DATA_DIR / "ecobici/stations/stations.csv"
    output_csv = DATA_DIR / "ecobici/stations/grid_1km.csv"

    create_1km_grid(input_csv, output_csv)
