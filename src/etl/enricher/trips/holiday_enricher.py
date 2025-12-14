import polars as pl
from pathlib import Path


class HolidayEnricher:
    """
    Enriquece los datos de viajes con información de días festivos.
    Une los datos de viajes con un conjunto de datos de días festivos basado en la fecha, agregando una bandera 'is_holiday'
    y potencialmente otros atributos de días festivos.
    """

    def __init__(self, holidays_path: str | Path):
        self.holidays_path = Path(holidays_path)
        self.holidays_df = self._load_holidays()

    def _load_holidays(self) -> pl.DataFrame:
        print(f"Cargando días festivos desde {self.holidays_path}...")
        try:
            df = pl.read_csv(self.holidays_path)
            # Parsear columna Date a tipo Date
            return df.with_columns(
                pl.col("date").str.strptime(pl.Date, "%Y-%m-%d").alias("date_key")
            ).select(["date_key", "holiday", "type"])
        except Exception as e:
            print(f"Error cargando días festivos: {e}")
            raise

    def enrich(self, df: pl.DataFrame) -> pl.DataFrame:
        # Crear clave de fecha desde ts_origin para union
        df = df.with_columns(pl.col("ts_origin").dt.date().alias("date_key"))

        # Unir con dias festivos
        enriched_df = df.join(self.holidays_df, on="date_key", how="left")

        # Crear columna is_holiday
        enriched_df = enriched_df.with_columns(
            pl.col("holiday").is_not_null().alias("is_holiday")
        )

        # Eliminar date_key temporal
        return enriched_df.drop(["date_key"])
