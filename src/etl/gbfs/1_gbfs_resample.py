"""
Remuestreo de datos GBFS a intervalos de 10 minutos con relleno de huecos entre días.
"""
import os
from pathlib import Path

import polars as pl
from pyprojroot import here

BASE_DIR = here()
LOG_DIR = Path(os.getenv("LOG_DIR", BASE_DIR / "log"))
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
INPUT_DIR = DATA_DIR / "ecobici/gbfs/1_clean"
OUTPUT_DIR = DATA_DIR / "ecobici/gbfs/2_train"
OUTPUT_PATH = OUTPUT_DIR / "0_gbfs_resampled.parquet"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuracion
RESAMPLE_INTERVAL = "10m"
MAX_GAP_FOR_FILL = "20m"  # Rellenar huecos <= 20 minutos con ultimo valor conocido
COLUMNS_TO_KEEP = ["station_code", "num_bikes_available", "capacity"]


def resample_station_data(station_df: pl.DataFrame) -> pl.DataFrame:
    """
    Remuestrea los datos de una sola estación a intervalos de 5 minutos en todos los períodos de tiempo.
    
    - Usa el último valor en cada intervalo de 5 minutos
    - Rellena huecos <= 30 minutos con el último valor conocido (puede cruzar límites de días)
    - Usa NaN para huecos > 30 minutos
    - Crea una cuadrícula completa de 5 minutos cubriendo todo el rango de tiempo
    """

    station_df = station_df.sort("snapshot_time")
    
    # Usar group_by_dynamic para remuestreo basado en tiempo, luego upsample para crear cuadricula completa
    resampled = (
        station_df.sort("snapshot_time")
        .group_by_dynamic("snapshot_time", every=RESAMPLE_INTERVAL)
        .agg([
            pl.col("num_bikes_available").last(),
            pl.col("capacity").last(),
        ])
        .upsample(time_column="snapshot_time", every=RESAMPLE_INTERVAL)
    )
    
    # Encontrar el ultimo timestamp valido para cada fila (para calcular huecos)
    resampled = resampled.with_columns([
        pl.when(pl.col("num_bikes_available").is_not_null())
        .then(pl.col("snapshot_time"))
        .forward_fill()
        .alias("last_valid_time")
    ])
    
    # Calcular duracion de hueco desde el ultimo punto de datos valido
    resampled = resampled.with_columns([
        (pl.col("snapshot_time") - pl.col("last_valid_time")).alias("gap_duration")
    ])
    
    # Rellenar hacia adelante temporalmente
    filled = resampled.with_columns([
        pl.col("num_bikes_available").fill_null(strategy="forward"),
        pl.col("capacity").fill_null(strategy="forward"),
    ])
    
    # Crear indicador de imputacion: True si el valor era originalmente null y fue rellenado
    result = resampled.with_columns([
        pl.col("num_bikes_available").is_null().alias("was_null")
    ])
    
    # Extraer duracion maxima de hueco desde configuracion
    max_gap_minutes = int(MAX_GAP_FOR_FILL.rstrip('m'))
    
    # Aplicar logica de relleno:
    # - num_bikes_available: forward fill solo si hueco <= MAX_GAP_FOR_FILL
    # - capacity: forward fill sin limite (siempre)
    result = result.with_columns([
        pl.when(
            (pl.col("gap_duration") <= pl.duration(minutes=max_gap_minutes)) | 
            (pl.col("num_bikes_available").is_not_null())
        )
        .then(filled["num_bikes_available"])
        .otherwise(None)
        .alias("num_bikes_available"),
        
        # Capacity siempre usa forward fill sin limite
        filled["capacity"].alias("capacity"),
    ])
    
    # Crear bandera de imputacion: 1 si el valor era null y fue rellenado (dentro de hueco de 30 min)
    result = result.with_columns([
        (pl.col("was_null") & pl.col("num_bikes_available").is_not_null()).cast(pl.Int8).alias("is_imputed")
    ])
    
    # Extraer hora y minuto para identificar periodo nocturno (00:30 - 05:00)
    result = result.with_columns([
        pl.col("snapshot_time").dt.hour().alias("hour"),
        pl.col("snapshot_time").dt.minute().alias("minute")
    ])
    
    # Identificar periodo nocturno: desde 00:30 hasta 04:59
    # (hora == 0 AND minuto >= 30) OR (hora >= 1 AND hora < 5)
    result = result.with_columns([
        (
            ((pl.col("hour") == 0) & (pl.col("minute") >= 30)) |
            ((pl.col("hour") >= 1) & (pl.col("hour") < 5))
        ).alias("is_night_period")
    ])
    
    # Guardar el ultimo valor de num_bikes_available antes del periodo nocturno
    # Para aplicar forward fill durante el periodo de desactivacion (00:30-05:00)
    result = result.with_columns([
        pl.when(~pl.col("is_night_period"))
        .then(pl.col("num_bikes_available"))
        .otherwise(None)
        .forward_fill()
        .alias("num_bikes_before_shutdown")
    ])
    
    # Durante el periodo nocturno: reemplazar num_bikes_available con el valor antes de desactivacion
    result = result.with_columns([
        pl.when(pl.col("is_night_period") & pl.col("num_bikes_before_shutdown").is_not_null())
        .then(pl.col("num_bikes_before_shutdown"))
        .otherwise(pl.col("num_bikes_available"))
        .alias("num_bikes_available")
    ])
    
    # Actualizar bandera de imputacion: en periodo nocturno con forward fill, marcar como imputado
    result = result.with_columns([
        pl.when(pl.col("is_night_period") & (pl.col("num_bikes_available") == pl.col("num_bikes_before_shutdown")))
        .then(1)
        .otherwise(pl.col("is_imputed"))
        .cast(pl.Int8)
        .alias("is_imputed")
    ])
    
    # Crear feature is_operating: 1 cuando el sistema esta operando (05:00 - 00:30)
    result = result.with_columns([
        (~pl.col("is_night_period")).cast(pl.Int8).alias("is_operating")
    ])
    
    # Limpiar columnas temporales
    result = result.select(["snapshot_time", "num_bikes_available", "capacity", "is_imputed", "is_operating"])
    
    return result


def process_parquet_files():
    """Procesa todos los archivos parquet y remuestrea a intervalos de 5 minutos con relleno de huecos entre días."""
    print(f"Escaneando archivos parquet en {INPUT_DIR}...")
    
    # Encontrar todos los archivos parquet
    parquet_files = sorted(list(INPUT_DIR.rglob("*.parquet")))
    print(f"Encontrados {len(parquet_files)} archivos parquet")
    
    # Leer todos los archivos y concatenar
    print("\nLeyendo todos los archivos parquet...")
    all_dfs = []
    for idx, file_path in enumerate(parquet_files, 1):
        print(f"  Leyendo archivo {idx}/{len(parquet_files)}: {file_path.name}")
        df = pl.read_parquet(file_path)
        all_dfs.append(df.select(["snapshot_time"] + COLUMNS_TO_KEEP))
    
    print("\nConcatenando archivos...")
    combined_df = pl.concat(all_dfs, how="diagonal_relaxed")
    print(f"Forma combinada: {combined_df.shape}")
    
    # Obtener numero de estaciones para logging
    n_stations = combined_df["station_code"].n_unique()
    print(f"Procesando {n_stations} estaciones a través de todos los períodos de tiempo...")
    
    # Optimizacion Polars: usar group_by en lugar de bucle
    # Procesar todas las estaciones en paralelo usando map_groups
    print("  Remuestreando datos (procesamiento paralelo por estacion)...")
    result = (
        combined_df
        .sort(["station_code", "snapshot_time"])
        .group_by("station_code", maintain_order=True)
        .map_groups(
            lambda df: resample_station_data(
                df.select(["snapshot_time", "num_bikes_available", "capacity"])
            ).with_columns([
                pl.lit(df["station_code"][0]).alias("station_code")
            ])
        )
    )
    
    print(f"Forma final remuestreada: {result.shape}")
    
    # Ordenar por station_code y snapshot_time para mejor organizacion
    result = result.sort(["station_code", "snapshot_time"])
    
    # Imprimir estadisticas sobre valores imputados
    print("\n" + "="*70)
    print("ESTADÍSTICAS DE IMPUTACIÓN")
    print("="*70)
    
    total_rows = result.shape[0]
    imputed_rows = result.filter(pl.col("is_imputed") == 1).shape[0]
    imputed_pct = (imputed_rows / total_rows * 100) if total_rows > 0 else 0
    
    print(f"Total filas: {total_rows:,}")
    print(f"Filas imputadas: {imputed_rows:,} ({imputed_pct:.2f}%)")
    print(f"Filas de datos originales: {total_rows - imputed_rows:,} ({100 - imputed_pct:.2f}%)")
    
    # Rango de fechas
    min_date = result["snapshot_time"].min()
    max_date = result["snapshot_time"].max()
    print(f"\nRango de fechas: {min_date} a {max_date}")
    
    # Encontrar e imprimir todos los huecos (dado que instantaneas GBFS contienen todas las estaciones, los huecos son globales)
    print("\n" + "="*70)
    print("ANÁLISIS DE HUECOS (Huecos de Recolección GBFS)")
    print("="*70)
    print("Nota: Dado que cada instantánea GBFS contiene todas las estaciones, los huecos afectan a todas las estaciones por igual.")
    
    # Obtener timestamps unicos e identificar huecos en la recoleccion de datos
    # Usar una sola estacion para identificar huecos (son los mismos para todas)
    first_station = result["station_code"][0]
    station_sample = result.filter(pl.col("station_code") == first_station)
    
    gaps_df = (
        station_sample
        .sort("snapshot_time")
        .with_columns([
            # Crear ID de hueco: cambia al transicionar hacia/desde estado imputado
            (pl.col("is_imputed") != pl.col("is_imputed").shift(1))
            .cum_sum()
            .alias("gap_id")
        ])
        .filter(pl.col("is_imputed") == 1)  # Solo ver filas imputadas
        .group_by("gap_id")
        .agg([
            pl.col("snapshot_time").min().alias("gap_start"),
            pl.col("snapshot_time").max().alias("gap_end"),
            pl.len().alias("gap_length_bins"),
        ])
        .with_columns([
            ((pl.col("gap_end") - pl.col("gap_start")).dt.total_minutes() + 5).alias("gap_duration_min")
        ])
        .sort("gap_start")
    )
    
    total_gaps = gaps_df.shape[0]
    print(f"\nNúmero total de huecos de recolección: {total_gaps}")
    
    if total_gaps > 0:
        # Estadisticas de duracion de huecos
        print("\nEstadísticas de duración de huecos:")
        print(f"  Mín: {gaps_df['gap_duration_min'].min()} minutos")
        print(f"  Máx: {gaps_df['gap_duration_min'].max()} minutos")
        print(f"  Media: {gaps_df['gap_duration_min'].mean():.2f} minutos")
        print(f"  Mediana: {gaps_df['gap_duration_min'].median()} minutos")
        
        # Imprimir todos los huecos
        print(f"\nTodos los huecos de recolección (mostrando todos los {total_gaps} huecos):")
        print(gaps_df.select(["gap_start", "gap_end", "gap_length_bins", "gap_duration_min"]))
        
        print(f"\nNota: Cada hueco afectó a todas las {n_stations} estaciones en el sistema.")
    else:
        print("¡No se encontraron huecos de recolección!")
    
    # Encontrar e imprimir huecos > 30 minutos (datos sin rellenar/faltantes)
    print("\n" + "="*70)
    print("ANÁLISIS DE HUECOS GRANDES (Huecos sin rellenar > 30 minutos)")
    print("="*70)
    
    unfilled_gaps_df = (
        station_sample
        .sort("snapshot_time")
        .with_columns([
            # Crear ID de hueco: cambia al transicionar hacia/desde estado null
            (pl.col("num_bikes_available").is_null() != pl.col("num_bikes_available").is_null().shift(1))
            .cum_sum()
            .alias("gap_id")
        ])
        .filter(pl.col("num_bikes_available").is_null())  # Solo ver filas null
        .group_by("gap_id")
        .agg([
            pl.col("snapshot_time").min().alias("gap_start"),
            pl.col("snapshot_time").max().alias("gap_end"),
            pl.len().alias("gap_length_bins"),
        ])
        .with_columns([
            ((pl.col("gap_end") - pl.col("gap_start")).dt.total_minutes() + 5).alias("gap_duration_min")
        ])
        .sort("gap_start")
    )
    
    total_unfilled = unfilled_gaps_df.shape[0]
    print(f"\nNúmero total de huecos sin rellenar (> 30 minutos): {total_unfilled}")
    
    if total_unfilled > 0:
        # Estadisticas de duracion de huecos grandes
        print("\nEstadísticas de duración de huecos sin rellenar:")
        print(f"  Mín: {unfilled_gaps_df['gap_duration_min'].min()} minutos")
        print(f"  Máx: {unfilled_gaps_df['gap_duration_min'].max()} minutos")
        print(f"  Media: {unfilled_gaps_df['gap_duration_min'].mean():.2f} minutos")
        print(f"  Mediana: {unfilled_gaps_df['gap_duration_min'].median()} minutos")
        
        # Imprimir todos los huecos grandes
        print(f"\nTodos los huecos sin rellenar:")
        print(unfilled_gaps_df.select(["gap_start", "gap_end", "gap_length_bins", "gap_duration_min"]))
        
        print(f"\nNota: Cada hueco sin rellenar afectó a todas las {n_stations} estaciones en el sistema.")
    else:
        print("¡No se encontraron huecos sin rellenar (todos los huecos fueron <= 30 minutos y fueron rellenados)!")
    
    # Escribir archivo de salida unico
    print(f"\n{'='*70}")
    print(f"Escribiendo archivo de salida: {OUTPUT_PATH}")
    result.write_parquet(OUTPUT_PATH)
    print(f"Guardado: {OUTPUT_PATH} ({result.shape[0]} filas, {result.shape[1]} columnas)")
    print("="*70)


if __name__ == "__main__":
    print(f"Directorio de entrada: {INPUT_DIR}")
    print(f"Directorio de salida: {OUTPUT_DIR}")
    print(f"Intervalo de remuestreo: {RESAMPLE_INTERVAL}")
    print(f"Hueco máximo para relleno hacia adelante: {MAX_GAP_FOR_FILL}")
    print(f"Columnas a mantener: {COLUMNS_TO_KEEP}\n")
    
    process_parquet_files()
    
    print("\nRemuestreo completado!")
