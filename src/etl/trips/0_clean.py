"""
Limpieza y conversión de archivos de viajes CSV a Parquet particionado.
"""
import os
from pathlib import Path

import polars as pl
from pyprojroot import here

BASE_DIR = here()
LOG_DIR = Path(os.getenv("LOG_DIR", BASE_DIR / "log"))
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))


def convert_trips_to_parquet(input_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(list(input_dir.rglob("*.csv")))

    if not csv_files:
        print(f"No se encontraron archivos CSV en {input_dir}")
        return

    print(f"Encontrados {len(csv_files)} archivos CSV para procesar.")

    for file_path in csv_files:
        print(f"Procesando {file_path.name}...")
        try:
            # Leer CSV
            # Leemos todas las columnas como Utf8 (String) inicialmente para manejar tipos mixtos de forma segura,
            # luego las convertimos. O dejamos que Polars infiera y arregle después.
            # Dado que "Edad_Usuario" puede ser float pero debe ser int, leer como float y luego convertir a int es más seguro,
            # o leer como string y limpiar.
            # Intentemos inferir primero, pero sobreescribir columnas específicas si es necesario.

            q = pl.scan_csv(file_path, ignore_errors=True)
            
            # Normalizar nombres de columnas - manejar variaciones como Ciclo_EstacionArribo vs Ciclo_Estacion_Arribo
            columns = q.collect_schema().names()
            rename_map = {}
            
            # Mapear variaciones comunes de nombres de columnas
            for col in columns:
                if col == "Ciclo_EstacionRetiro":
                    rename_map[col] = "Ciclo_Estacion_Retiro"
                elif col == "Ciclo_EstacionArribo":
                    rename_map[col] = "Ciclo_Estacion_Arribo"
            
            if rename_map:
                q = q.rename(rename_map)

            # Renombrar columnas a nombres de esquema objetivo para facilitar la manipulación
            # Pero primero necesitamos procesar fechas usando nombres originales

            # Lógica para parsear fechas con formatos mixtos
            # Formatos: DD/MM/YYYY, YYYY-MM-DD, o DD/MM/YY
            # Nota: %y parsea año de 2 dígitos. Polars/chrono usualmente pivotea alrededor de 1969/1970 o 2000.
            # Para Ecobici (datos recientes), años de 2 dígitos como '24' deben mapearse a '2024'.

            # Helper para corregir años de 2 dígitos si se parsean como 0022 en lugar de 2022
            # Esto sucede si el formato no pivotea automáticamente correctamente o si los datos son solo "22"

            def parse_date_col(col_name):
                return (
                    pl.col(col_name)
                    .str.strptime(pl.Date, "%d/%m/%Y", strict=False)
                    .fill_null(
                        pl.col(col_name).str.strptime(pl.Date, "%Y-%m-%d", strict=False)
                    )
                    .fill_null(
                        pl.col(col_name).str.strptime(pl.Date, "%d/%m/%y", strict=False)
                    )
                )

            fecha_retiro_expr = parse_date_col("Fecha_Retiro")
            fecha_arribo_expr = parse_date_col("Fecha_Arribo")  # Parseo de tiempo
            # Manejar tiempos con microsegundos (ej., 07:03:42.510000) tomando solo los primeros 8 caracteres (HH:MM:SS)
            # o usando un formato que acepte microsegundos.
            # Ya que se pidió descartar la última parte:

            hora_retiro_expr = (
                pl.col("Hora_Retiro")
                .str.slice(0, 8)  # Mantener solo HH:MM:SS
                .str.strptime(pl.Time, "%H:%M:%S", strict=False)
            )

            hora_arribo_expr = (
                pl.col("Hora_Arribo")
                .str.slice(0, 8)  # Mantener solo HH:MM:SS
                .str.strptime(pl.Time, "%H:%M:%S", strict=False)
            )

            # Transformaciones
            df = (
                q.with_columns(
                    [
                        # Convertir Edad a Int (manejar floats convirtiendo a Float64 primero luego Int64, o solo Int64 si está limpio)
                        # El usuario dijo "A veces es un float".
                        pl.col("Edad_Usuario")
                        .cast(pl.Float64, strict=False)
                        .cast(pl.Int64, strict=False)
                        .alias("user_age"),
                        # Convertir Bici a Int
                        pl.col("Bici").cast(pl.Int64, strict=False).alias("bicycle_id"),
                        # Renombrar columnas simples
                        pl.col("Genero_Usuario").alias("user_gender"),
                        pl.col("Ciclo_Estacion_Retiro")
                        .cast(pl.Utf8)
                        .alias("station_id_origin"),
                        pl.col("Ciclo_Estacion_Arribo")
                        .cast(pl.Utf8)
                        .alias("station_id_destination"),
                        # Crear Timestamps
                        fecha_retiro_expr.alias("_date_origin"),
                        hora_retiro_expr.alias("_time_origin"),
                        fecha_arribo_expr.alias("_date_dest"),
                        hora_arribo_expr.alias("_time_dest"),
                    ]
                )
                .with_columns(
                    [
                        pl.col("_date_origin")
                        .dt.combine(pl.col("_time_origin"))
                        .dt.replace_time_zone(
                            "America/Mexico_City", ambiguous="earliest"
                        )
                        .alias("ts_origin"),
                        pl.col("_date_dest")
                        .dt.combine(pl.col("_time_dest"))
                        .dt.replace_time_zone(
                            "America/Mexico_City", ambiguous="earliest"
                        )
                        .alias("ts_destination"),
                    ]
                )
                .with_columns(
                    (
                        (
                            pl.col("ts_destination") - pl.col("ts_origin")
                        ).dt.total_seconds()
                        / 60
                    )
                    .round(0)
                    .cast(pl.Int64)
                    .alias("trip_duration_minutes")
                )
                .select(
                    [
                        "station_id_origin",
                        "station_id_destination",
                        "ts_origin",
                        "ts_destination",
                        "trip_duration_minutes",
                        "bicycle_id",
                        "user_gender",
                        "user_age",
                    ]
                )
                .sort("ts_origin")
                .collect()
            )

            # Escribir a Parquet con particionamiento estilo Hive
            # Necesitamos extraer año y mes de ts_origin para particionar
            # Corregir año si es < 100 (ej. 22 -> 2022)
            df = df.with_columns(
                [
                    pl.col("ts_origin").dt.year().alias("year"),
                    pl.col("ts_origin").dt.month().alias("month"),
                ]
            ).with_columns(
                pl.when(pl.col("year") < 100)
                .then(pl.col("year") + 2000)
                .otherwise(pl.col("year"))
                .alias("year")
            )

            # Aplicar filtros de limpieza antes de escribir
            # Limpiar IDs de estación
            df = df.with_columns(
                [
                    pl.col("station_id_origin").str.replace(r"\.0+$", ""),
                    pl.col("station_id_destination").str.replace(r"\.0+$", ""),
                ]
            )

            # Filtrar viajes
            hour = pl.col("ts_origin").dt.hour()
            minute = pl.col("ts_origin").dt.minute()

            # Excluir viajes entre 00:30 y 05:00
            exclude_time_condition = ((hour == 0) & (minute >= 30)) | (
                (hour >= 1) & (hour < 5)
            )

            initial_count = df.height
            df = df.filter(
                (pl.col("trip_duration_minutes") <= 60)
                & (pl.col("station_id_origin") != pl.col("station_id_destination"))
                & (pl.col("station_id_origin").str.contains(r"^\d+(-\d+)?$"))
                & (pl.col("station_id_destination").str.contains(r"^\d+(-\d+)?$"))
                & (~exclude_time_condition)
            )

            # Eliminar duplicados
            df = df.unique()

            final_count = df.height
            print(f"  Limpieza: {initial_count} -> {final_count} filas (eliminadas {initial_count - final_count})")

            if final_count == 0:
                print(f"  Saltando guardado (vacío después de la limpieza)")
                continue

            # Escribir dataset particionado
            base_filename = file_path.stem

            # Particionar por año y mes
            partitions = df.select(["year", "month"]).unique()

            for row in partitions.iter_rows(named=True):
                y = row["year"]
                m = row["month"]

                if y is None or m is None:
                    continue

                partition_dir = output_dir / f"year={y}" / f"month={m}"
                partition_dir.mkdir(parents=True, exist_ok=True)

                # Filtrar datos para esta partición
                partition_df = df.filter(
                    (pl.col("year") == y) & (pl.col("month") == m)
                ).drop(["year", "month"])

                output_file = partition_dir / f"{base_filename}.parquet"
                partition_df.write_parquet(output_file)

            print(f"  Procesado y limpiado {file_path.name} en particiones.")

        except Exception as e:
            print(f"  Error procesando {file_path.name}: {e}")


if __name__ == "__main__":
    input_dir = DATA_DIR / "ecobici/trips/0_raw"
    output_dir = DATA_DIR / "ecobici/trips/1_clean"

    print(f"Input Dir: {input_dir}")
    print(f"Output Dir: {output_dir}")
    
    convert_trips_to_parquet(input_dir, output_dir)
