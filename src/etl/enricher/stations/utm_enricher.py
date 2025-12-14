"""
Convierte coordenadas geográficas (WGS84) a coordenadas proyectadas UTM.
"""
import polars as pl
from pyproj import Transformer


class UTMEnricher:
    """
    Convierte coordenadas geográficas (WGS84) a coordenadas proyectadas UTM.
    
    Para Ciudad de México se utiliza la zona UTM 14N (EPSG:32614).
    
    Las coordenadas UTM son más adecuadas para modelos de ML porque:
    - Están en metros (escala física real)
    - Son coordenadas cartesianas (no esféricas)
    - Permiten calcular distancias euclidianas directamente
    """

    def __init__(self, utm_zone: str = "32614"):
        """
        Args:
            utm_zone: Código EPSG de la zona UTM. 
                     Por defecto 32614 = UTM Zone 14N (Ciudad de México)
        """
        self.utm_zone = utm_zone
        # Crear transformador de WGS84 (EPSG:4326) a UTM
        self.transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_zone}", always_xy=True)

    def enrich(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Añade columnas utm_x y utm_y al DataFrame.
        
        Args:
            df: DataFrame con columnas 'latitude' y 'longitude'
            
        Returns:
            DataFrame con las columnas adicionales 'utm_x' y 'utm_y'
        """
        # Detectar nombres de columnas (case-insensitive)
        lat_col = "latitude" if "latitude" in df.columns else "Latitude"
        lon_col = "longitude" if "longitude" in df.columns else "Longitude"

        if lat_col not in df.columns or lon_col not in df.columns:
            print(f"Columnas {lat_col}/{lon_col} no encontradas. Saltando UTM enrichment.")
            return df

        print(f"Convirtiendo WGS84 → UTM {self.utm_zone}...")

        # Convertir a pandas para pyproj (más rápido que iterar)
        coords = df.select([lon_col, lat_col]).to_pandas()
        
        # Transformar (pyproj espera lon, lat para always_xy=True)
        utm_x, utm_y = self.transformer.transform(
            coords[lon_col].values, 
            coords[lat_col].values
        )

        # Añadir nuevas columnas
        df = df.with_columns([
            pl.Series("utm_x", utm_x),
            pl.Series("utm_y", utm_y)
        ])

        print(f"  Anadidas columnas utm_x, utm_y")
        
        return df
