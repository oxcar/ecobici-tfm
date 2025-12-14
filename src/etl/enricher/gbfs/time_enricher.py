"""
Enriquecedor que agrega características de tiempo cíclico a los datos GBFS.
"""
import numpy as np
import polars as pl


class CyclicTimeEnricher:
    """
    Enriquecedor que agrega características de tiempo cíclico a los datos GBFS.
    
    Crea codificaciones de seno y coseno para:
    - Hora del día (0-23)
    - Día de la semana (1-7, Lunes=1)
    - Mes (1-12)
    """
    
    def __init__(self, time_column: str = "snapshot_time"):
        self.time_column = time_column
    
    def enrich(self, df: pl.DataFrame) -> pl.DataFrame:
        """Agrega características de tiempo cíclico al dataframe."""
        
        # Extraer componentes de tiempo
        df = df.with_columns([
            pl.col(self.time_column).dt.hour().alias("hour"),
            pl.col(self.time_column).dt.minute().alias("minute"),
            pl.col(self.time_column).dt.weekday().alias("weekday"),  # Lunes=1, Domingo=7
            pl.col(self.time_column).dt.month().alias("month"),
        ])
        
        # Calcular minuto del dia (0-1439)
        df = df.with_columns([
            (pl.col("hour") * 60 + pl.col("minute")).alias("minute_of_day")
        ])
        
        # Crear codificaciones ciclicas
        # Tiempo (ciclo de 1440 minutos por dia)
        # Argumento = 2pi x (minuto_del_dia / 1440)
        df = df.with_columns([
            (2 * np.pi * pl.col("minute_of_day") / 1440).sin().alias("time_sin"),
            (2 * np.pi * pl.col("minute_of_day") / 1440).cos().alias("time_cos"),
        ])
        
        # Dia de la semana (ciclo de 7 dias)
        # Usar weekday - 1 para convertir 1-7 a 0-6 solo para el calculo
        # Argumento = 2pi x ((dia_semana - 1) / 7)
        df = df.with_columns([
            (2 * np.pi * (pl.col("weekday") - 1) / 7).sin().alias("day_sin"),
            (2 * np.pi * (pl.col("weekday") - 1) / 7).cos().alias("day_cos"),
        ])
        
        # Mes (ciclo de 12 meses)
        # Argumento = 2pi x ((month - 1) / 12)
        df = df.with_columns([
            (2 * np.pi * (pl.col("month") - 1) / 12).sin().alias("month_sin"),
            (2 * np.pi * (pl.col("month") - 1) / 12).cos().alias("month_cos"),
        ])
        
        # Indicador de fin de semana (Sabado=6, Domingo=7)
        df = df.with_columns([
            (pl.col("weekday") >= 6).cast(pl.Int8).alias("is_weekend")
        ])
        
        # Eliminar columnas intermedias
        df = df.drop(["minute", "minute_of_day"])
        
        return df
