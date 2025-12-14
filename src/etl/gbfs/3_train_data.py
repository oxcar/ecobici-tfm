
"""
Realiza los pasos finales de limpieza y guardado de los datos de entrenamiento GBFS.
"""
import os
import polars as pl
from pathlib import Path
from pyprojroot import here

BASE_DIR = here()
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
INPUT_DIR = DATA_DIR / "ecobici/gbfs/2_train"
OUTPUT_DIR = INPUT_DIR

input_path: Path = INPUT_DIR / "1_gbfs_train.parquet"
output_path: Path = OUTPUT_DIR / "2_gbfs_train.parquet"

def finalize_gbfs_train_data(input_path: Path, output_path: Path):
    """
    Realiza pasos finales de limpieza y guardado de los datos de entrenamiento GBFS.
    """
    
    if not input_path.exists():
        print(f"{input_path} no existe.")
        return
    
    print(f"Leyendo {input_path.name}")
    df = pl.read_parquet(input_path)
    print(f"{df.shape[0]:,} filas, {df.shape[1]} columnas")
    
    # Validar datos antes de calcular ocupacion
    print("\nValidando datos...")
    
    # Identificar registros con valores anormales
    anomalies = df.filter(
        (pl.col("bikes") > pl.col("capacity")) |
        (pl.col("bikes") < 0) |
        (pl.col("capacity") <= 0)
    )
    
    if anomalies.shape[0] > 0:
        print(f"ADVERTENCIA: Se encontraron {anomalies.shape[0]:,} registros con valores anormales")
        print(f"  - bikes > capacity o bikes < 0 o capacity <= 0")
        print(f"  Estos registros seran corregidos:")
        print(f"    - bikes se limitara al rango [0, capacity]")
    
    # Corregir valores anormales: limitar bikes al rango [0, capacity]
    df = df.with_columns([
        pl.when(pl.col("bikes") > pl.col("capacity"))
          .then(pl.col("capacity"))
          .when(pl.col("bikes") < 0)
          .then(0)
          .otherwise(pl.col("bikes"))
          .alias("bikes"),
        
        pl.when(pl.col("bikes_lag_1") > pl.col("capacity"))
          .then(pl.col("capacity"))
          .when(pl.col("bikes_lag_1") < 0)
          .then(0)
          .otherwise(pl.col("bikes_lag_1"))
          .alias("bikes_lag_1"),
        
        pl.when(pl.col("bikes_lag_2") > pl.col("capacity"))
          .then(pl.col("capacity"))
          .when(pl.col("bikes_lag_2") < 0)
          .then(0)
          .otherwise(pl.col("bikes_lag_2"))
          .alias("bikes_lag_2"),
        
        pl.when(pl.col("bikes_lag_6") > pl.col("capacity"))
          .then(pl.col("capacity"))
          .when(pl.col("bikes_lag_6") < 0)
          .then(0)
          .otherwise(pl.col("bikes_lag_6"))
          .alias("bikes_lag_6"),
        
        pl.when(pl.col("bikes_lag_12") > pl.col("capacity"))
          .then(pl.col("capacity"))
          .when(pl.col("bikes_lag_12") < 0)
          .then(0)
          .otherwise(pl.col("bikes_lag_12"))
          .alias("bikes_lag_12"),
        
        pl.when(pl.col("bikes_lag_138") > pl.col("capacity"))
          .then(pl.col("capacity"))
          .when(pl.col("bikes_lag_138") < 0)
          .then(0)
          .otherwise(pl.col("bikes_lag_138"))
          .alias("bikes_lag_138"),
        
        pl.when(pl.col("bikes_lag_144") > pl.col("capacity"))
          .then(pl.col("capacity"))
          .when(pl.col("bikes_lag_144") < 0)
          .then(0)
          .otherwise(pl.col("bikes_lag_144"))
          .alias("bikes_lag_144"),
        
        pl.when(pl.col("target_bikes_20min") > pl.col("capacity"))
          .then(pl.col("capacity"))
          .when(pl.col("target_bikes_20min") < 0)
          .then(0)
          .otherwise(pl.col("target_bikes_20min"))
          .alias("target_bikes_20min"),
        
        pl.when(pl.col("target_bikes_40min") > pl.col("capacity"))
          .then(pl.col("capacity"))
          .when(pl.col("target_bikes_40min") < 0)
          .then(0)
          .otherwise(pl.col("target_bikes_40min"))
          .alias("target_bikes_40min"),
        
        pl.when(pl.col("target_bikes_60min") > pl.col("capacity"))
          .then(pl.col("capacity"))
          .when(pl.col("target_bikes_60min") < 0)
          .then(0)
          .otherwise(pl.col("target_bikes_60min"))
          .alias("target_bikes_60min"),
    ])
    
    # Calcular ocupacion y deltas de ocupacion
    print("\nCalculando metricas de ocupacion...")
    df = df.with_columns([
        # Ocupacion actual y lags
        (pl.col("bikes") / pl.col("capacity")).alias("ocu"),
        (pl.col("bikes_lag_1") / pl.col("capacity")).alias("ocu_lag_1"),
        (pl.col("bikes_lag_2") / pl.col("capacity")).alias("ocu_lag_2"),
        (pl.col("bikes_lag_6") / pl.col("capacity")).alias("ocu_lag_6"),
        (pl.col("bikes_lag_12") / pl.col("capacity")).alias("ocu_lag_12"),
        (pl.col("bikes_lag_138") / pl.col("capacity")).alias("ocu_lag_138"),
        (pl.col("bikes_lag_144") / pl.col("capacity")).alias("ocu_lag_144"),
        
        # Trends de ocupacion (cambio desde cada lag)
        (pl.col("bikes_trend_1") / pl.col("capacity")).alias("ocu_trend_1"),
        (pl.col("bikes_trend_2") / pl.col("capacity")).alias("ocu_trend_2"),
        (pl.col("bikes_trend_6") / pl.col("capacity")).alias("ocu_trend_6"),
        (pl.col("bikes_trend_12") / pl.col("capacity")).alias("ocu_trend_12"),
        
        # Deltas de ocupacion (cambio futuro respecto a actual)
        ((pl.col("target_bikes_20min") / pl.col("capacity")) - (pl.col("bikes") / pl.col("capacity"))).alias("ocu_delta_20"),
        ((pl.col("target_bikes_40min") / pl.col("capacity")) - (pl.col("bikes") / pl.col("capacity"))).alias("ocu_delta_40"),
        ((pl.col("target_bikes_60min") / pl.col("capacity")) - (pl.col("bikes") / pl.col("capacity"))).alias("ocu_delta_60"),
    ])
    
    # Eliminar columnas originales de bikes y trends
    print("Eliminando columnas originales de bikes y trends")
    df = df.drop([
        "bikes",
        "bikes_lag_1",
        "bikes_lag_2",
        "bikes_lag_6",
        "bikes_lag_12",
        "bikes_lag_138",
        "bikes_lag_144",
        "bikes_trend_1",
        "bikes_trend_2",
        "bikes_trend_6",
        "bikes_trend_12",
        "bikes_trend_138",
        "bikes_trend_144",
        "target_bikes_20min",
        "target_bikes_40min",
        "target_bikes_60min"
    ])

    print("Eliminando otras columnas")
    df = df.drop([
        "month",
        "month_sin",
        "month_cos",
        "latitude",
        "longitude",
        "apparent_temperature",
        "occupancy",  # Corregido typo de 'occupancy',
    ])
    
    print(f"\nForma final: {df.shape}")
    
    print(f"\nGuardando datos limpios en {output_path}...")
    df.write_parquet(output_path)
    print("Guardado completado.")

if __name__ == "__main__":
    print(f"Archivo de entrada: {input_path}")
    print(f"Archivo de salida: {output_path}\n")
    
    finalize_gbfs_train_data(input_path, output_path)