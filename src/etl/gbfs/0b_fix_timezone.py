"""
Script para corregir la zona horaria en archivos parquet GBFS existentes.
Los timestamps están guardados en UTC pero deberían interpretarse como hora local de Ciudad de México.
"""
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
from pyprojroot import here

BASE_DIR = here()
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
PARQUET_DIR = DATA_DIR / "ecobici/gbfs/1_clean"
OUTPUT_DIR = DATA_DIR / "ecobici/gbfs/1b_clean"
TIMEZONE = ZoneInfo(os.getenv("TIMEZONE", "America/Mexico_City"))


def fix_timezone_in_parquet(file_path: Path, output_path: Path):
    """
    Lee un archivo parquet, corrige la zona horaria de snapshot_time y lo guarda.
    
    Los timestamps están en UTC pero representan hora local de Ciudad de México.
    Necesitamos:
    1. Quitar la zona horaria (strip timezone)
    2. Aplicar la zona horaria de Ciudad de México
    """
    print(f"Procesando: {file_path}")
    
    # Leer parquet
    df = pl.read_parquet(file_path)
    
    # Verificar si tiene la columna snapshot_time
    if "snapshot_time" not in df.columns:
        print(f"  No tiene columna snapshot_time, saltando...")
        return
    
    # Corregir timezone
    df = df.with_columns([
        pl.col("snapshot_time")
        .dt.replace_time_zone(None)  # Quitar timezone (tratar como naive)
        .dt.replace_time_zone(str(TIMEZONE))  # Aplicar timezone de Ciudad de México
        .alias("snapshot_time")
    ])
    
    # Crear directorio de salida
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Guardar
    df.write_parquet(output_path)
    print(f"  Guardado en: {output_path}")


def main():
    print("Corrigiendo zona horaria en archivos GBFS parquet...")
    print(f"Directorio de entrada: {PARQUET_DIR}")
    print(f"Directorio de salida: {OUTPUT_DIR}")
    print()
    
    # Encontrar todos los archivos parquet
    parquet_files = list(PARQUET_DIR.rglob("*.parquet"))
    
    if not parquet_files:
        print("No se encontraron archivos parquet")
        return
    
    print(f"Encontrados {len(parquet_files)} archivos parquet")
    print()
    
    for file_path in sorted(parquet_files):
        # Calcular ruta relativa para mantener estructura
        rel_path = file_path.relative_to(PARQUET_DIR)
        output_path = OUTPUT_DIR / rel_path
        
        try:
            fix_timezone_in_parquet(file_path, output_path)
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    print()
    print("Proceso completado")


if __name__ == "__main__":
    main()
