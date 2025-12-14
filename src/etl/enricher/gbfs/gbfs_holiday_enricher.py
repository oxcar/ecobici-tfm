"""
Enriquece los datos GBFS con información de días festivos.
"""
import polars as pl
from pathlib import Path


class GBFSHolidayEnricher:
    """
    Enriquece los datos GBFS con información de días festivos.
    Une los datos GBFS con un conjunto de datos de días festivos basado en la fecha, agregando una bandera 'is_holiday'
    y potencialmente otros atributos de días festivos.
    """

    def __init__(self, holidays_path: str | Path):
        self.holidays_path = Path(holidays_path)
        self.holidays_df = self._load_holidays()

    def _load_holidays(self) -> pl.DataFrame:
        print(f"  Cargando días festivos desde {self.holidays_path}...")
        try:
            df = pl.read_csv(self.holidays_path)
            # Parsear columna date a tipo Date
            df = df.with_columns(
                pl.col("date").str.strptime(pl.Date, "%Y-%m-%d").alias("date_key")
            )
            
            # Seleccionar columnas relevantes
            return df.select(["date_key", "holiday", "type"])
        except Exception as e:
            print(f"  Error cargando días festivos: {e}")
            raise

    def enrich(self, df: pl.DataFrame) -> pl.DataFrame:
        """Agrega información de días festivos al dataframe GBFS."""
        
        # Crear clave de fecha desde snapshot_time para union
        df = df.with_columns(
            pl.col("snapshot_time").dt.date().alias("date_key")
        )

        # Unir con dias festivos para verificar si la fecha existe en el dataset de festivos
        df = df.join(
            self.holidays_df.select(["date_key"]).with_columns([
                pl.lit(1).cast(pl.Int8).alias("is_holiday")
            ]),
            on="date_key",
            how="left"
        )

        # Rellenar valores null en is_holiday con 0
        df = df.with_columns(
            pl.col("is_holiday").fill_null(0).cast(pl.Int8)
        )

        # Eliminar columna temporal date_key
        df = df.drop(["date_key"])
        
        return df
