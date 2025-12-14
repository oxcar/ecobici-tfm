"""
Enriquecedor que añade características temporales de actividad de estaciones a los datos GBFS.
"""
import polars as pl
from pathlib import Path


class StationActivityEnricher:
    """
    Enriquecedor que añade características temporales de actividad de estaciones a los datos GBFS.
    
    Une los datos GBFS con características de actividad calculadas por (station_code, hour, weekday):
    - station_netflow: Flujo neto promedio (arrivals - departures)
    - station_intensity: Intensidad de actividad (arrivals + departures)
    """
    
    def __init__(self, activity_path: str | Path):
        """
        Inicializa el enriquecedor de actividad de estaciones.
        
        Args:
            activity_path: Ruta al archivo parquet de características de actividad de estaciones
        """
        self.activity_path = Path(activity_path)
        self._activity_data = None
    
    def _load_activity_data(self) -> pl.DataFrame:
        """Carga datos de actividad de estaciones."""
        if self._activity_data is not None:
            return self._activity_data
        
        print(f"  Cargando características de actividad de estaciones desde {self.activity_path}...")
        
        if not self.activity_path.exists():
            print(f"  Advertencia: Archivo de actividad no encontrado en {self.activity_path}")
            return pl.DataFrame()
        
        # Seleccionar solo columnas necesarias para union
        required_cols = [
            "station_code",
            "hour",
            "weekday",
            "station_netflow_rate",
            "station_turnover_rate",
        ]
        
        df = pl.scan_parquet(self.activity_path).select(required_cols).collect()
        
        # Verificar que todas las columnas existen
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  Advertencia: Faltan columnas en archivo de actividad: {missing_cols}")
            return pl.DataFrame()
        
        print(f"  Cargadas {df.shape[0]} combinaciones (estación, hora, día_semana)")
        
        self._activity_data = df
        return self._activity_data
    
    def enrich(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Añade características de actividad de estaciones al dataframe GBFS.
        
        Une por (station_code, hour, weekday).
        """
        activity_df = self._load_activity_data()
        
        if activity_df.is_empty():
            print("  Advertencia: No hay datos de actividad disponibles, omitiendo enriquecimiento de actividad")
            return df
        
        print(f"  Uniendo características de actividad de estaciones...")
        
        # Extraer hour y weekday de snapshot_time si no existen
        needs_hour = "hour" not in df.columns
        needs_weekday = "weekday" not in df.columns
        
        if needs_hour or needs_weekday:
            temp_cols = []
            if needs_hour:
                temp_cols.append(pl.col("snapshot_time").dt.hour().alias("hour"))
            if needs_weekday:
                temp_cols.append((pl.col("snapshot_time").dt.weekday() - 1).alias("weekday"))
            
            df = df.with_columns(temp_cols)
        
        # Asegurar que hour y weekday sean del mismo tipo que en activity_df
        df = df.with_columns([
            pl.col("hour").cast(pl.Int8),
            pl.col("weekday").cast(pl.Int8)
        ])
        
        # Convertir a LazyFrame para union mas eficiente
        df_lazy = df.lazy()
        activity_lazy = activity_df.lazy()
        
        # Unir con datos de actividad usando streaming
        df_lazy = df_lazy.join(
            activity_lazy,
            on=["station_code", "hour", "weekday"],
            how="left"
        )
        
        # Recolectar resultado con motor streaming
        df = df_lazy.collect()
        
        # Reportar cobertura
        null_count = df.filter(pl.col("station_netflow_rate").is_null()).shape[0]
        total = df.shape[0]
        matched = total - null_count
        print(f"    Uniones exitosas: {matched:,}/{total:,} ({matched/total*100:.1f}%)")
        
        if null_count > 0:
            print(f"    Advertencia: {null_count:,} filas sin características de actividad (combinación estación/hora/día no encontrada)")
        
        return df
