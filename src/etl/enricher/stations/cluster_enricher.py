"""
Enriquece los datos de estaciones con la columna de cluster.
Lee los clusters desde un archivo CSV y los une por station_code.
"""
from pathlib import Path

import polars as pl


class ClusterEnricher:
    """
    Enriquece los datos de estaciones con la columna de cluster.
    Lee los clusters desde un archivo CSV y los une por station_code.
    """

    def __init__(self, clusters_csv_path: str | Path):
        self.clusters_path = Path(clusters_csv_path)
        self.clusters_df = self._load_clusters()

    def _load_clusters(self) -> pl.DataFrame:
        print(f"Cargando clusters desde {self.clusters_path}...")
        
        if not self.clusters_path.exists():
            raise FileNotFoundError(f"Archivo de clusters no encontrado: {self.clusters_path}")
        
        df = pl.read_csv(
            self.clusters_path,
            schema_overrides={"station_code": pl.String, "cluster": pl.Int32}
        )
        
        # Validar que existan las columnas necesarias
        required_cols = ["station_code", "cluster"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Columnas faltantes en archivo de clusters: {missing_cols}")
        
        # Mantener solo las columnas necesarias
        df = df.select(["station_code", "cluster"])
        
        print(f"  - {len(df)} estaciones con cluster cargadas")
        return df

    def enrich(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Agrega la columna cluster al DataFrame de estaciones.
        """
        if "station_code" not in df.columns:
            raise ValueError("DataFrame debe contener columna 'station_code'")
        
        # Unir por station_code
        df_enriched = df.join(
            self.clusters_df,
            on="station_code",
            how="left"
        )
        
        # Contar estaciones sin cluster asignado
        null_count = df_enriched.filter(pl.col("cluster").is_null()).height
        if null_count > 0:
            print(f"  - Advertencia: {null_count} estaciones sin cluster asignado")
        
        print(f"  - Columna 'cluster' agregada")
        return df_enriched
