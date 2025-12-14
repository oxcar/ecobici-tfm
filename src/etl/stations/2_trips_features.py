"""
Genera características de actividad de viajes por estación basadas en datos históricos de viajes.
Calcula el flujo neto y la intensidad de uso por estación, hora del día y día de la semana.
"""
import os
from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import datetime
import calendar

import polars as pl
from pyprojroot import here

BASE_DIR = here()
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
TRIPS_DIR = DATA_DIR / "ecobici/trips/1_clean"
STATIONS_FILE = DATA_DIR / "ecobici/stations/0_stations.csv"
OUTPUT_FILE = DATA_DIR / "ecobici/stations/2_stations_activity_features.parquet"
TIMEZONE = ZoneInfo(os.getenv("TIMEZONE", "America/Mexico_City"))

# Configuracion: numero de meses hacia atras para buscar datos de viajes
LOOKBACK_MONTHS = int(os.getenv("TRIPS_LOOKBACK_MONTHS", "6"))

def get_latest_month_from_trips() -> tuple[str, str]:
    """Obtiene el rango de fechas para analizar viajes (lookback de N meses)."""
    print(f"Buscando último mes disponible en {TRIPS_DIR}...")
    print(f"Lookback configurado: {LOOKBACK_MONTHS} meses")
    
    # Buscar todas las carpetas year=XXXX/month=XX
    year_dirs = sorted(
        [d for d in TRIPS_DIR.glob("year=*") if d.is_dir()],
        key=lambda x: int(x.name.split("=")[1]),
        reverse=True
    )
    
    if not year_dirs:
        raise FileNotFoundError(f"No se encontraron directorios year= en {TRIPS_DIR}")
    
    # Tomar el último año
    latest_year_dir = year_dirs[0]
    latest_year = latest_year_dir.name.split("=")[1]
    
    # Buscar todos los meses en ese año
    month_dirs = sorted(
        [d for d in latest_year_dir.glob("month=*") if d.is_dir()],
        key=lambda x: int(x.name.split("=")[1]),
        reverse=True
    )
    
    if not month_dirs:
        raise FileNotFoundError(f"No se encontraron directorios month= en {latest_year_dir}")
    
    # Tomar el último mes como fecha final
    latest_month_dir = month_dirs[0]
    latest_month = latest_month_dir.name.split("=")[1]
    
    year_int = int(latest_year)
    month_int = int(latest_month)
    
    # Calcular último día del mes final
    last_day = calendar.monthrange(year_int, month_int)[1]
    end_date_obj = datetime(year_int, month_int, last_day)
    end_date = end_date_obj.strftime("%Y-%m-%d")
    
    # Calcular fecha de inicio restando LOOKBACK_MONTHS meses
    # Ir hacia atrás mes por mes
    start_year = year_int
    start_month = month_int - LOOKBACK_MONTHS + 1  # +1 porque incluimos el mes actual
    
    # Ajustar año si el mes es negativo o cero
    while start_month <= 0:
        start_month += 12
        start_year -= 1
    
    start_date = f"{start_year}-{start_month:02d}-01"
    
    print(f"Último mes encontrado: {latest_year}/{latest_month}")
    print(f"Rango de fechas: {start_date} a {end_date} ({LOOKBACK_MONTHS} meses)")
    
    return start_date, end_date


def calculate_netflow_features(trips_dir: Path, date_range: tuple[str, str]) -> pl.DataFrame:
    """Calcula características de flujo neto por estación."""
    print(f"Calculando características de flujo neto desde {trips_dir}...")
    
    start_date, end_date = date_range
    
    # Cargar viajes filtrados por rango de fechas
    lazy_df = pl.scan_parquet(str(trips_dir / "**/*.parquet"))
    lazy_df = lazy_df.filter(
        (pl.col("ts_origin").dt.date() >= pl.lit(start_date).str.strptime(pl.Date, "%Y-%m-%d")) &
        (pl.col("ts_origin").dt.date() <= pl.lit(end_date).str.strptime(pl.Date, "%Y-%m-%d"))
    )
    
    # Calcular salidas por estación, hora, día de semana y fecha
    print("\tCalculando salidas por día...")
    departures = (
        lazy_df
        .with_columns([
            pl.col("ts_origin").dt.date().alias("date"),
            pl.col("ts_origin").dt.hour().alias("hour"),
            pl.col("ts_origin").dt.weekday().alias("weekday"),
        ])
        .group_by(["station_id_origin", "date", "hour", "weekday"])
        .agg([pl.len().alias("departures")])
        .rename({"station_id_origin": "station_code"})
        .collect()
    )
    
    # Calcular llegadas por estación, hora, día de semana y fecha
    print("  Calculando llegadas por día...")
    arrivals = (
        lazy_df
        .with_columns([
            pl.col("ts_destination").dt.date().alias("date"),
            pl.col("ts_destination").dt.hour().alias("hour"),
            pl.col("ts_destination").dt.weekday().alias("weekday"),
        ])
        .group_by(["station_id_destination", "date", "hour", "weekday"])
        .agg([pl.len().alias("arrivals")])
        .rename({"station_id_destination": "station_code"})
        .collect()
    )
    
    # Unir salidas y llegadas por fecha
    print("  Calculando flujo neto diario...")
    netflow_daily = departures.join(
        arrivals,
        on=["station_code", "date", "hour", "weekday"],
        how="full",
        coalesce=True
    ).with_columns([
        pl.col("departures").fill_null(0),
        pl.col("arrivals").fill_null(0)
    ])
    
    # Calcular promedios por combinación de hora y día de la semana (promediando los días)
    print("  Calculando promedios por hora y día de semana...")
    netflow_avg = (
        netflow_daily
        .group_by(["station_code", "hour", "weekday"])
        .agg([
            pl.col("arrivals").mean().alias("station_arrivals"),
            pl.col("departures").mean().alias("station_departures"),
        ])
        .with_columns([
            (pl.col("station_arrivals") - pl.col("station_departures")).alias("station_netflow"),
            (pl.col("station_arrivals") + pl.col("station_departures")).alias("station_intensity")
        ])
    )
    
    print(f"Calculado para {netflow_avg['station_code'].n_unique()} estaciones")
    
    return netflow_avg


def create_5min_intervals(date_range: tuple[str, str]) -> pl.DataFrame:
    """Crea intervalos de 5 minutos para el rango de fechas dado."""
    print("Creando intervalos de 5 minutos...")
    
    start_date, end_date = date_range
    
    # Crear rango de timestamps cada 5 minutos
    timestamps = pl.datetime_range(
        start=pl.datetime(int(start_date[:4]), int(start_date[5:7]), int(start_date[8:])),
        end=pl.datetime(int(end_date[:4]), int(end_date[5:7]), int(end_date[8:]), 23, 55),
        interval="5m",
        eager=True
    )
    
    df = pl.DataFrame({"timestamp": timestamps})
    
    # Aplicar zona horaria y extraer hora y dia de semana
    df = df.with_columns([
        pl.col("timestamp").dt.replace_time_zone(str(TIMEZONE)).alias("timestamp")
    ]).with_columns([
        pl.col("timestamp").dt.hour().alias("hour"),
        pl.col("timestamp").dt.weekday().alias("weekday")
    ])
    
    print(f"  Creados {len(df)} intervalos de 5 minutos")
    return df


def build_station_features(netflow_data: pl.DataFrame, intervals_df: pl.DataFrame, stations: list[str]) -> pl.DataFrame:
    """Construye el dataframe final con características por estación e intervalo."""
    print(f"Construyendo características para {len(stations)} estaciones...")
    
    # Crear el producto cartesiano: estaciones x timestamps
    all_combinations = []
    for station in stations:
        station_df = intervals_df.with_columns([
            pl.lit(station).alias("station_code")
        ])
        all_combinations.append(station_df)
    
    result = pl.concat(all_combinations)
    
    # Unir con netflow usando hora y dia de semana
    result = result.join(
        netflow_data,
        on=["station_code", "hour", "weekday"],
        how="left"
    )
    
    # Rellenar nulls con 0 (usando coalesce para evitar problemas de tipo)
    result = result.with_columns([
        pl.coalesce([pl.col("station_arrivals"), pl.lit(0.0)]).alias("station_arrivals"),
        pl.coalesce([pl.col("station_departures"), pl.lit(0.0)]).alias("station_departures"),
        pl.coalesce([pl.col("station_netflow"), pl.lit(0.0)]).alias("station_netflow"),
        pl.coalesce([pl.col("station_intensity"), pl.lit(0.0)]).alias("station_intensity")
    ])
    
    # Agrupar y agregar por station_code, hour, weekday descartando timestamp
    print("  Agrupando por estación, hora y día de semana...")
    result = result.group_by(["station_code", "hour", "weekday"]).agg([
        pl.col("station_netflow").mean().alias("station_netflow"),
        pl.col("station_intensity").mean().alias("station_intensity")
    ])
    
    # Seleccionar columnas finales
    result = result.select([
        "station_code",
        "hour",
        "weekday",
        "station_netflow",
        "station_intensity"
    ])
    
    # Ordenar por station_code, hour, weekday
    result = result.sort(["station_code", "hour", "weekday"])
    
    print(f"  DataFrame final: {len(result)} filas")
    return result


def main():
    print("Generando características de actividad de viajes por estación...")
    
    # 1. Cargar estaciones autorizadas con capacidad
    print(f"Cargando estaciones autorizadas desde {STATIONS_FILE}...")
    stations_df = pl.read_csv(STATIONS_FILE).select(["station_code", "capacity"])
    authorized_stations = (
        stations_df
        .select("station_code")
        .unique()
        .to_series()
        .to_list()
    )
    print(f"\tEstaciones autorizadas: {len(authorized_stations)}")
    
    # 2. Obtener último mes disponible en viajes
    date_range = get_latest_month_from_trips()
    
    # 3. Calcular caracteristicas de flujo neto
    netflow_data = calculate_netflow_features(TRIPS_DIR, date_range)
    
    # 4. Filtrar solo estaciones autorizadas
    netflow_data = netflow_data.filter(
        pl.col("station_code").is_in(authorized_stations)
    )
    
    # 5. Usar lista de estaciones autorizadas
    stations = sorted(authorized_stations)
    print(f"Estaciones a procesar: {len(stations)}")
    
    # 6. Crear intervalos de 5 minutos
    intervals_df = create_5min_intervals(date_range)
    
    # 7. Construir dataframe final
    result = build_station_features(netflow_data, intervals_df, stations)
    
    # 8. Unir con capacidad de estaciones y calcular station_turnover
    print("Calculando station_turnover...")
    result = result.join(
        stations_df,
        on="station_code",
        how="left"
    )
    
    result = result.with_columns([
        (pl.col("station_intensity") / pl.col("capacity")).alias("station_turnover_rate"),
        (pl.col("station_netflow") / pl.col("capacity")).alias("station_netflow_rate")
    ])
    
    # Seleccionar columnas finales en el orden deseado
    result = result.select([
        "station_code",
        "weekday",
        "hour",
        # "station_netflow",
        # "station_intensity",
        # "capacity",
        "station_turnover_rate",
        "station_netflow_rate"
    ])
    
    print(f"\tstation_turnover calculado para {result['station_code'].n_unique()} estaciones")
    
    # 9. Ordenar antes de guardar
    print("Ordenando datos por station_code, weekday y hour...")
    result = result.sort(["station_code", "weekday", "hour"])
    
    # 10. Guardar a parquet
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(OUTPUT_FILE)
    print(f"\nGuardado en: {OUTPUT_FILE}")
    print(f"Tamaño del archivo: {OUTPUT_FILE.stat().st_size / (1024*1024):.2f} MB")


if __name__ == "__main__":
    main()
