"""
Enriquecedor que añade características estáticas de estaciones a los datos GBFS.
"""
import polars as pl
from pathlib import Path


class StationEnricher:
    """
    Enriquecedor que añade características estáticas de estaciones a los datos GBFS.
    
    Une los datos GBFS con metadatos enriquecidos de estaciones incluyendo:
    - Características geográficas (lat, lon)
    - Características de PDI (comercio, finanzas, cultura, etc.)
    - Características de tránsito (metro, metrobús, etc.)
    - Características socioeconómicas (IDS)
    - Características de actividad de la estación (ratio de flujo, intensidad, patrones pico)
    """
    
    def __init__(self, stations_path: str | Path):
        """
        Inicializa el enriquecedor estático de estaciones.
        
        Args:
            stations_path: Ruta al archivo parquet de estaciones enriquecidas
        """
        self.stations_path = Path(stations_path)
        self._stations_data = None
    
    def _load_stations_data(self) -> pl.DataFrame:
        """Carga datos enriquecidos de estaciones."""
        if self._stations_data is not None:
            return self._stations_data
        
        print(f"  Cargando características de estaciones desde {self.stations_path}...")
        
        if not self.stations_path.exists():
            print(f"  Advertencia: Archivo de estaciones no encontrado en {self.stations_path}")
            return pl.DataFrame()
        
        df = pl.read_parquet(self.stations_path)
        
        # Eliminar station_id ya que usaremos station_code para unir
        # También eliminar name y capacity ya que pueden existir en GBFS
        cols_to_drop = ["station_id"]
        if "name" in df.columns:
            cols_to_drop.append("name")
        if "capacity" in df.columns:
            cols_to_drop.append("capacity")
        
        df = df.drop(cols_to_drop)
        
        print(f"  Cargadas {df.shape[0]} estaciones con {df.shape[1]} características")
        
        self._stations_data = df
        return self._stations_data
    
    def enrich(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Añade características estáticas de estaciones al dataframe GBFS.
        
        Une por station_code.
        """
        stations_df = self._load_stations_data()
        
        if stations_df.is_empty():
            print("  Advertencia: No hay datos de estaciones disponibles, omitiendo enriquecimiento de estaciones")
            return df
        
        print(f"  Uniendo características de estaciones...")
        
        # Unir con datos de estaciones
        df = df.join(
            stations_df,
            on="station_code",
            how="left"
        )
        
        # Reportar cobertura
        # Verificar si la latitud es nula (como proxy para unión exitosa)
        if "latitude" in df.columns:
            null_count = df.filter(pl.col("latitude").is_null()).shape[0]
            total = df.shape[0]
            coverage = (total - null_count) / total * 100 if total > 0 else 0
            print(f"  Cobertura de características de estaciones: {coverage:.2f}% ({total - null_count:,}/{total:,} filas)")
        
        return df
