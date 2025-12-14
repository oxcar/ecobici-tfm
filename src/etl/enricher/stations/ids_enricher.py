"""
Enriquece los datos de estaciones con información sociodemográfica (IDS - Índice de Desarrollo Social).
Utiliza un BallTree para encontrar puntos de datos IDS dentro de radios de 300m y 500m de cada estación
y agrega valores de población e índice de desarrollo social.
"""
import json
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.neighbors import BallTree


class IDSEnricher:
    """
    Enriquece los datos de estaciones con información sociodemográfica (IDS - Índice de Desarrollo Social).
    Utiliza un BallTree para encontrar puntos de datos IDS dentro de radios de 300m y 500m de cada estación
    y agrega valores de población e índice de desarrollo social.
    """

    def __init__(self, geojson_path: str | Path):
        self.geojson_path = Path(geojson_path)
        self.data_values, self.coords = self._load_data()
        self.tree = BallTree(self.coords, metric="haversine")

    def _load_data(self):
        print(f"Cargando datos IDS desde {self.geojson_path}...")
        with open(self.geojson_path, "r") as f:
            data = json.load(f)

        coords = []
        values = []

        for feature in data.get("features", []):
            props = feature.get("properties", {})
            geom = feature.get("geometry", {})
            if geom.get("type") == "Point":
                lon, lat = geom.get("coordinates")
                coords.append([lat, lon])
                values.append(
                    {
                        "ids_pobtotal": props.get("pobtotal"),
                        "ids_pobres_tot": props.get("pobres_tot"),
                        "ids_ids": props.get("ids"),
                    }
                )

        if not coords:
            raise ValueError(f"No se encontraron datos válidos en {self.geojson_path}")

        return values, np.radians(np.array(coords))

    def enrich(self, df: pl.DataFrame) -> pl.DataFrame:
        lat_col = "latitude" if "latitude" in df.columns else "Latitude"
        lon_col = "longitude" if "longitude" in df.columns else "Longitude"

        if lat_col not in df.columns or lon_col not in df.columns:
            print("Advertencia: columnas lat/lon no encontradas para enriquecimiento IDS")
            return df

        station_lats = np.radians(df[lat_col].to_numpy())
        station_lons = np.radians(df[lon_col].to_numpy())
        station_coords = np.vstack([station_lats, station_lons]).T

        # Radio de la Tierra en metros
        R = 6371000
        r300 = 300 / R

        # Consulta de radio
        inds_300 = self.tree.query_radius(station_coords, r=r300)

        def aggregate(indices_list, values_list, key, func, default=None):
            result = []
            for indices in indices_list:
                if len(indices) == 0:
                    result.append(default)
                else:
                    vals = [
                        values_list[i][key]
                        for i in indices
                        if values_list[i][key] is not None
                    ]
                    if not vals:
                        result.append(default)
                    else:
                        result.append(func(vals))
            return result

        # Agregaciones 300m
        pobtotal_300 = aggregate(inds_300, self.data_values, "ids_pobtotal", sum, default=0)
        ids_300 = aggregate(inds_300, self.data_values, "ids_ids", np.mean, default=None)

        # Calcular mediana global de ids_300m para valores validos
        valid_ids = [v for v in ids_300 if v is not None]
        ids_global_median = np.median(valid_ids) if valid_ids else 0.0

        # Imputar ids_300m con mediana global cuando poblacion es 0
        ids_300_imputed = [
            ids_global_median if pop == 0 else ids
            for pop, ids in zip(pobtotal_300, ids_300)
        ]

        return df.with_columns(
            [
                pl.Series("ids_population_300m", pobtotal_300),
                pl.Series("ids_300m", ids_300_imputed),
            ]
        )
