"""
Enriquecedor que agrega datos meteorológicos a las instantáneas GBFS.
"""
import polars as pl
from pathlib import Path


class WeatherEnricher:
    """
    Enriquecedor que agrega datos meteorológicos a las instantáneas GBFS.
    
    Los datos meteorológicos tienen resolución horaria. Los datos GBFS tienen resolución de 5 minutos.
    Este enriquecedor unirá los datos meteorológicos a la hora más cercana para cada instantánea GBFS.
    """
    
    def __init__(self, weather_dir: str | Path):
        """
        Inicializa el enriquecedor meteorológico.
        
        Args:
            weather_dir: Directorio que contiene archivos parquet meteorológicos en formato:
                        station=XXX/weather.parquet
        """
        self.weather_dir = Path(weather_dir)
        self._weather_data = None
    
    def _load_weather_data(self) -> pl.DataFrame:
        """Carga todos los datos meteorológicos de los directorios de estaciones."""
        if self._weather_data is not None:
            return self._weather_data
        
        print(f"  Cargando datos meteorológicos desde {self.weather_dir}...")
        
        # Encontrar todos los archivos weather.parquet
        weather_files = list(self.weather_dir.rglob("weather.parquet"))
        
        if not weather_files:
            print(f"  Advertencia: No se encontraron archivos meteorológicos en {self.weather_dir}")
            return pl.DataFrame()
        
        print(f"  Encontrados {len(weather_files)} archivos meteorológicos")
        
        # Leer todos los archivos meteorologicos y extraer station_code del nombre de directorio
        dfs = []
        for file in weather_files:
            df = pl.read_parquet(file)
            
            # Extraer codigo de estacion del path: station=XXX/weather.parquet -> XXX
            station_code = file.parent.name.replace("station=", "")
            
            # Agregar columna station_code
            df = df.with_columns([
                pl.lit(station_code).alias("station_code")
            ])
            
            dfs.append(df)
        
        # Concatenar todos los datos meteorologicos
        weather_df = pl.concat(dfs, how="diagonal_relaxed")
        
        # Convertir columna time a datetime si es string y agregar zona horaria
        if weather_df["time"].dtype == pl.String or weather_df["time"].dtype == pl.Utf8:
            weather_df = weather_df.with_columns([
                pl.col("time").str.to_datetime().dt.replace_time_zone("America/Mexico_City")
            ])
        elif weather_df["time"].dtype.time_zone is None:
            # Si datetime pero sin zona horaria, agregar zona horaria de Ciudad de Mexico
            weather_df = weather_df.with_columns([
                pl.col("time").dt.replace_time_zone("America/Mexico_City")
            ])
        
        # Filtrar solo datos 2025 desde Marzo en adelante
        weather_df = weather_df.filter(
            (pl.col("time").dt.year() >= 2025) &
            (pl.col("time").dt.month() >= 3)
        )
        
        print(f"  Cargados {weather_df.shape[0]:,} registros meteorológicos (Marzo 2025+)")
        
        self._weather_data = weather_df
        return self._weather_data
    
    def enrich(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Agrega datos meteorológicos al dataframe GBFS.
        
        Los datos meteorológicos son horarios, así que redondearemos cada snapshot_time a la hora más cercana
        y uniremos con los datos meteorológicos en (station_code, hour).
        """
        weather_df = self._load_weather_data()
        
        if weather_df.is_empty():
            print("  Advertencia: No hay datos meteorológicos disponibles, omitiendo enriquecimiento meteorológico")
            return df
        
        # Redondear snapshot_time a la hora mas cercana para union
        df = df.with_columns([
            pl.col("snapshot_time").dt.truncate("1h").alias("weather_hour")
        ])
        
        # Unir con datos meteorologicos
        print(f"  Uniendo datos meteorológicos...")
        df = df.join(
            weather_df,
            left_on=["station_code", "weather_hour"],
            right_on=["station_code", "time"],
            how="left"
        )
        
        # Eliminar columna temporal weather_hour, columna 'time' unida, y weather_code
        cols_to_drop = ["weather_hour"]
        if "time" in df.columns:
            cols_to_drop.append("time")
        if "weather_code" in df.columns:
            cols_to_drop.append("weather_code")
        
        df = df.drop(cols_to_drop)
        
        # Reportar datos meteorologicos faltantes (usar apparent_temperature en lugar de temperature_2m)
        null_weather = df.filter(pl.col("apparent_temperature").is_null()).shape[0]
        total = df.shape[0]
        coverage = (total - null_weather) / total * 100 if total > 0 else 0
        
        print(f"  Cobertura meteorológica: {coverage:.2f}% ({total - null_weather:,}/{total:,} filas)")
        
        return df
