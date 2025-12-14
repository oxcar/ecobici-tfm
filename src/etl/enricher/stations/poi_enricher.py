"""
Enriquece los datos de estaciones con métricas de proximidad a Puntos de Interés (POI).
"""
import json
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.neighbors import BallTree


class POIEnricher:
    """
    Enriquece los datos de estaciones con métricas de proximidad a Puntos de Interés (POI).
    Calcula el número de POIs de una categoría específica dentro de 300m y 500m,
    la distancia al POI más cercano, y la distancia media a los k POIs más cercanos.
    """

    def __init__(self, poi_geojson_path: str | Path, category_name: str):
        self.poi_path = Path(poi_geojson_path)
        self.category = category_name
        self.poi_coords = self._load_pois()

        # Construir BallTree para consultas eficientes
        # La métrica 'haversine' espera radianes
        self.tree = BallTree(self.poi_coords, metric="haversine")

    def _load_pois(self) -> np.ndarray:
        print(f"Cargando POIs desde {self.poi_path}...")
        with open(self.poi_path, "r") as f:
            data = json.load(f)

        coords = []
        for feature in data.get("features", []):
            geom = feature.get("geometry", {})
            if geom.get("type") == "Point":
                # GeoJSON es [lon, lat]
                lon, lat = geom.get("coordinates")
                coords.append(
                    [lat, lon]
                )  # BallTree espera [lat, lon] para haversine si convertimos a radianes

        if not coords:
            raise ValueError(f"No se encontraron puntos en {self.poi_path}")

        # Convertir a radianes para haversine
        return np.radians(np.array(coords))

    def enrich(self, df: pl.DataFrame) -> pl.DataFrame:
        # Detectar columnas lat/lon
        lat_col = "latitude" if "latitude" in df.columns else "Latitude"
        lon_col = "longitude" if "longitude" in df.columns else "Longitude"

        if lat_col not in df.columns or lon_col not in df.columns:
            print(
                f"Advertencia: No se pudieron encontrar columnas lat/lon en el dataframe de estaciones. Columnas: {df.columns}"
            )
            return df

        # Obtener coordenadas de estaciones en radianes
        station_lats = np.radians(df[lat_col].to_numpy())
        station_lons = np.radians(df[lon_col].to_numpy())
        station_coords = np.vstack([station_lats, station_lons]).T

        # Radio de la Tierra en metros
        R = 6371000

        # 1. Número de POIs a 300m
        radius_300 = 300 / R
        count_300 = self.tree.query_radius(
            station_coords, r=radius_300, count_only=True
        )

        # # 2. Número de POIs a 500m
        # radius_500 = 500 / R
        # count_500 = self.tree.query_radius(
        #     station_coords, r=radius_500, count_only=True
        # )

        # # 3. Distancia al POI más cercano
        # # k=1 devuelve distancias e índices
        # dist_nearest, _ = self.tree.query(station_coords, k=1)
        # dist_nearest_m = dist_nearest.flatten() * R

        # # 4. Distancia media a los k POIs más cercanos
        # k = 3
        # # Manejar caso donde hay menos de 10 POIs
        # k_actual = min(k, len(self.poi_coords))
        # dist_k, _ = self.tree.query(station_coords, k=k_actual)
        # mean_dist_k_m = np.mean(dist_k, axis=1) * R

        # Agregar columnas al DataFrame
        return df.with_columns(
            [
                pl.Series(f"{self.category}_pois_300m", count_300),
                # pl.Series(f"{self.category}_pois_500m", count_500),
                # pl.Series(f"{self.category}_nearest_dist_m", dist_nearest_m),
                # pl.Series(f"{self.category}_mean_dist_{k}_m", mean_dist_k_m),
            ]
        )
