"""
Enriquece los datos de estaciones con métricas de proximidad al transporte público.
"""
from pathlib import Path
import numpy as np
import polars as pl
import json
from sklearn.neighbors import BallTree


class TransitEnricher:
    """
    Enriquece los datos de estaciones con métricas de proximidad al transporte público.
    Combina todas las categorías de transporte (Metro, Metrobús, etc.) en un único conjunto de métricas:
    - transit_nearest_station_m: Distancia a la estación de transporte más cercana de cualquier tipo
    - transit_stations_300m: Conteo de todas las estaciones de transporte dentro de 300m
    """
    def __init__(self, geojson_paths: list[str | Path]):
        self.geojson_paths = [Path(p) for p in geojson_paths]
        self.coords = self._load_all_data()
        self.tree = BallTree(self.coords, metric="haversine")

    def _load_all_data(self) -> np.ndarray:
        """Carga y combina todas las coordenadas de estaciones de transporte desde múltiples archivos GeoJSON."""
        all_coords = []
        
        for geojson_path in self.geojson_paths:
            print(f"Cargando datos de transporte desde {geojson_path}...")
            with open(geojson_path, "r") as f:
                data = json.load(f)

            for feature in data.get("features", []):
                geom = feature.get("geometry", {})
                if geom.get("type") == "Point":
                    coordinates = geom.get("coordinates")
                    if coordinates and len(coordinates) >= 2:
                        lon, lat = coordinates[:2]
                        all_coords.append([lat, lon])

        if not all_coords:
            raise ValueError(f"No se encontraron puntos válidos en ningún archivo GeoJSON de transporte")

        print(f"Cargadas {len(all_coords)} estaciones de transporte totales desde {len(self.geojson_paths)} archivos")
        return np.radians(np.array(all_coords))

    def enrich(self, df: pl.DataFrame) -> pl.DataFrame:
        lat_col = "latitude" if "latitude" in df.columns else "Latitude"
        lon_col = "longitude" if "longitude" in df.columns else "Longitude"

        if lat_col not in df.columns or lon_col not in df.columns:
            print(
                f"Advertencia: columnas lat/lon no encontradas para enriquecimiento de transporte"
            )
            return df

        station_lats = np.radians(df[lat_col].to_numpy())
        station_lons = np.radians(df[lon_col].to_numpy())
        station_coords = np.vstack([station_lats, station_lons]).T

        R = 6371000

        # 1. Distancia a la estacion de transporte mas cercana
        dist_nearest, _ = self.tree.query(station_coords, k=1)
        dist_nearest_m = dist_nearest.flatten() * R

        # 2. Conteo de estaciones de transporte dentro de 300m
        radius_300 = 300 / R
        count_300 = self.tree.query_radius(
            station_coords, r=radius_300, count_only=True
        )

        return df.with_columns(
            [
                pl.Series("transit_nearest_station_m", dist_nearest_m),
                pl.Series("transit_stations_300m", count_300),
            ]
        )
