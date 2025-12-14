"""
Utilidades compartidas para los modelos de prediccion de ocupacion Ecobici.

Este modulo contiene funciones comunes para:
- Carga de datos
- Division temporal por semanas
- Calculo de metricas (MAE, RMSE)
- Visualizaciones estandarizadas
"""

import numpy as np
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from typing import Tuple, Dict, List, Optional

# =============================================================================
# Constantes
# =============================================================================

# Ruta al dataset (relativa desde los notebooks en src/modeler/ via symbolic link)
DATA_PATH = 'data/ecobici/gbfs/2_train/2_gbfs_train.parquet'

# Features para los modelos
FEATURE_COLS = [
    # Ocupacion actual y lags
    'ocu',
    'ocu_lag_1',
    'ocu_lag_2',
    'ocu_lag_6',
    'ocu_lag_12',
    'ocu_lag_138',
    'ocu_lag_144',
    # Features de tendencia
    'ocu_trend_1',   # cambio en ultimos 10 min
    'ocu_trend_2',   # cambio en ultimos 20 min
    'ocu_trend_6',   # cambio en ultima hora
    'ocu_trend_12',  # cambio en ultimas 2 horas
    # Temporales ciclicas
    'time_sin',
    'time_cos',
    'day_sin',
    'day_cos',
    'is_weekend',
    'is_holiday',
    # Capacidad y estado operativo
    'capacity',
    'is_operating',
    # POIs cercanos
    'commerce_pois_300m',
    'finance_pois_300m',
    'culture_pois_300m',
    'education_pois_300m',
    'sport_recreation_pois_300m',
    'hotels_pois_300m',
    'food_pois_300m',
    'health_pois_300m',
    'drink_pois_300m',
    # Transporte publico
    'transit_nearest_station_m',
    'transit_stations_300m',
    # Demografia
    'ids_population_300m',
    'ids_300m',
    # Ubicacion
    'utm_x',
    'utm_y',
    # Flujo de estacion
    'station_netflow_rate',
    'station_turnover_rate',
    # Meteorologia
    'temperature_2m',
    'rain',
    'surface_pressure',
    'cloud_cover',
    'wind_speed_10m',
    'relative_humidity_2m',
]

# Todas las features (incluye trends que vienen en el dataset)
BASE_FEATURE_COLS = FEATURE_COLS

# Variables objetivo (deltas de ocupacion)
TARGETS = ['ocu_delta_20', 'ocu_delta_40', 'ocu_delta_60']

# Horizontes de prediccion en minutos
HORIZONS = [20, 40, 60]

# Configuracion de division de datos (semanas)
# Total: 30 semanas exactas (multiplos de semanas completas)
TOTAL_WEEKS = 30
TRAIN_WEEKS = 22
VAL_WEEKS = 4
TEST_WEEKS = 4


# =============================================================================
# Funciones de carga de datos
# =============================================================================

def load_data(data_path: str = DATA_PATH, extra_cols: List[str] = None, filter_extra_nulls: bool = True) -> pd.DataFrame:
    """
    Carga el dataset de entrenamiento desde parquet usando DuckDB.
    
    El dataset incluye features de tendencia precalculadas:
    - ocu_trend_1, ocu_trend_2, ocu_trend_6, ocu_trend_12
    
    Parametros
    ----------
    data_path : str
        Ruta al archivo parquet
    extra_cols : List[str], optional
        Columnas adicionales a cargar (por ejemplo, 'cluster')
    filter_extra_nulls : bool, default True
        Si True, filtra registros con valores nulos en extra_cols
        
    Retorna
    -------
    pd.DataFrame
        DataFrame con los datos filtrados (sin nulos en targets ni features)
    """
    con = duckdb.connect()
    
    # Columnas a cargar (sin las de tendencia que se calculan al vuelo)
    columns = ['snapshot_time', 'station_code'] + BASE_FEATURE_COLS + TARGETS
    
    # Agregar columnas extra si se especifican
    if extra_cols:
        columns = columns + [c for c in extra_cols if c not in columns]
    
    # Construir condiciones NOT NULL para todas las columnas relevantes
    not_null_conditions = [f"{col} IS NOT NULL" for col in BASE_FEATURE_COLS + TARGETS]
    
    # Agregar condiciones NOT NULL para columnas extra solo si se solicita
    if extra_cols and filter_extra_nulls:
        not_null_conditions += [f"{col} IS NOT NULL" for col in extra_cols]
    
    where_clause = " AND ".join(not_null_conditions)
    
    query = f"""
        SELECT {', '.join(columns)}
        FROM parquet_scan('{data_path}')
        WHERE {where_clause}
        ORDER BY station_code, snapshot_time
    """
    
    df = con.sql(query).df()
    con.close()
    
    print(f"Datos cargados: {len(df):,} registros")
    print(f"Estaciones: {df['station_code'].nunique()}")
    print(f"Periodo: {df['snapshot_time'].min()} a {df['snapshot_time'].max()}")
    print(f"Features incluyen lags y trends precalculados: ocu_lag_1,2,6,12,138,144 y ocu_trend_1,2,6,12")
    
    # Verificar que no hay NaN
    nan_count = df[FEATURE_COLS + TARGETS].isna().sum().sum()
    if nan_count > 0:
        print(f"Advertencia: Se encontraron {nan_count} valores NaN despues del filtrado")
    else:
        print("Verificacion: No hay valores NaN en features ni targets")
    
    return df


def split_data_by_weeks(
    df: pd.DataFrame,
    total_weeks: int = TOTAL_WEEKS,
    train_weeks: int = TRAIN_WEEKS,
    val_weeks: int = VAL_WEEKS,
    test_weeks: int = TEST_WEEKS
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide los datos cronologicamente por semanas ISO.
    
    La division se hace por semanas completas para evitar sobrerrepresentacion
    de ciertos dias de la semana y respetar la estacionalidad semanal.
    
    - Test: ultimas 4 semanas del dataset
    - Validacion: 4 semanas anteriores a test
    - Entrenamiento: 22 semanas anteriores a validacion
    
    Se eliminan registros del inicio para asegurar exactamente 30 semanas.
    
    Parametros
    ----------
    df : pd.DataFrame
        DataFrame con columna 'snapshot_time'
    total_weeks : int
        Numero total de semanas a usar (default: 30)
    train_weeks : int
        Numero de semanas para entrenamiento (default: 22)
    val_weeks : int
        Numero de semanas para validacion (default: 4)
    test_weeks : int
        Numero de semanas para pruebas (default: 4)
        
    Retorna
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        DataFrames de train, validacion y test
    """
    # Zona horaria de Ciudad de Mexico
    MEXICO_TZ = 'America/Mexico_City'
    
    # Crear columna de semana ISO (ano-semana) usando zona horaria de CDMX
    df = df.copy()
    
    # Convertir a zona horaria de Mexico si tiene timezone, o localizar si es naive
    if df['snapshot_time'].dt.tz is None:
        snapshot_local = df['snapshot_time'].dt.tz_localize(MEXICO_TZ)
    else:
        snapshot_local = df['snapshot_time'].dt.tz_convert(MEXICO_TZ)
    
    df['iso_week'] = snapshot_local.dt.isocalendar().week
    df['iso_year'] = snapshot_local.dt.isocalendar().year
    df['year_week'] = df['iso_year'].astype(str) + '-W' + df['iso_week'].astype(str).str.zfill(2)
    
    # Obtener semanas unicas ordenadas cronologicamente
    weeks = df.sort_values('snapshot_time')['year_week'].unique()
    total_available = len(weeks)
    
    print(f"Total de semanas disponibles en el dataset: {total_available}")
    print(f"Zona horaria utilizada: {MEXICO_TZ}")
    print(f"Primera semana: {weeks[0]}")
    print(f"Ultima semana: {weeks[-1]}")
    
    # Verificar que hay suficientes semanas
    required_weeks = train_weeks + val_weeks + test_weeks
    if required_weeks != total_weeks:
        raise ValueError(
            f"La suma de semanas no coincide: {train_weeks} + {val_weeks} + {test_weeks} = {required_weeks} != {total_weeks}"
        )
    
    if total_available < total_weeks:
        raise ValueError(
            f"No hay suficientes semanas. Requeridas: {total_weeks}, Disponibles: {total_available}"
        )
    
    # Seleccionar las ultimas total_weeks semanas (eliminar semanas del inicio)
    weeks_to_use = weeks[-total_weeks:]
    weeks_removed = total_available - total_weeks
    
    if weeks_removed > 0:
        print(f"\nSemanas eliminadas del inicio: {weeks_removed}")
        print(f"  Primera semana eliminada: {weeks[0]}")
        print(f"  Ultima semana eliminada: {weeks[weeks_removed - 1]}")
    
    print(f"\nSemanas a utilizar: {total_weeks} (de {weeks_to_use[0]} a {weeks_to_use[-1]})")
    
    # Dividir semanas desde el final (test al final, train al inicio)
    # Las ultimas test_weeks son para test
    test_week_list = weeks_to_use[-test_weeks:]
    # Las val_weeks anteriores son para validacion
    val_week_list = weeks_to_use[-(test_weeks + val_weeks):-test_weeks]
    # El resto son para entrenamiento
    train_week_list = weeks_to_use[:train_weeks]
    
    print(f"\nSemanas asignadas:")
    print(f"  Train: {train_week_list[0]} a {train_week_list[-1]} ({len(train_week_list)} semanas)")
    print(f"  Val:   {val_week_list[0]} a {val_week_list[-1]} ({len(val_week_list)} semanas)")
    print(f"  Test:  {test_week_list[0]} a {test_week_list[-1]} ({len(test_week_list)} semanas)")
    
    # Filtrar datos
    train_df = df[df['year_week'].isin(train_week_list)].copy()
    val_df = df[df['year_week'].isin(val_week_list)].copy()
    test_df = df[df['year_week'].isin(test_week_list)].copy()
    
    # Eliminar columnas auxiliares
    for d in [train_df, val_df, test_df]:
        d.drop(columns=['iso_week', 'iso_year', 'year_week'], inplace=True)
    
    print(f"\nDivision de datos:")
    print(f"  Entrenamiento: {len(train_df):,} registros ({train_weeks} semanas)")
    print(f"  Validacion:    {len(val_df):,} registros ({val_weeks} semanas)")
    print(f"  Pruebas:       {len(test_df):,} registros ({test_weeks} semanas)")
    
    # Mostrar rangos de fechas
    print(f"\nRangos de fechas:")
    print(f"  Train: {train_df['snapshot_time'].min()} a {train_df['snapshot_time'].max()}")
    print(f"  Val:   {val_df['snapshot_time'].min()} a {val_df['snapshot_time'].max()}")
    print(f"  Test:  {test_df['snapshot_time'].min()} a {test_df['snapshot_time'].max()}")
    
    return train_df, val_df, test_df


# =============================================================================
# Funciones de metricas
# =============================================================================

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, capacity: np.ndarray = None, baseline_mae: float = None) -> Dict[str, float]:
    """
    Calcula MAE, RMSE, MAE en bicicletas y Skill Score.
    
    Parametros
    ----------
    y_true : np.ndarray
        Valores reales (delta de ocupacion)
    y_pred : np.ndarray
        Valores predichos (delta de ocupacion)
    capacity : np.ndarray, optional
        Capacidad de cada estacion para calcular MAE en bicicletas
    baseline_mae : float, optional
        MAE del modelo baseline para calcular Skill Score
        
    Retorna
    -------
    Dict[str, float]
        Diccionario con MAE, RMSE, MAE_bikes (si capacity es proporcionado) y SS (si baseline_mae es proporcionado)
    """
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    result = {
        'MAE': mae,
        'RMSE': rmse
    }
    
    # MAE en bicicletas (si se proporciona capacidad)
    if capacity is not None:
        # Error en bicicletas = |delta_true - delta_pred| * capacidad
        mae_bikes = np.mean(np.abs(y_true - y_pred) * capacity)
        result['MAE_bikes'] = mae_bikes
    
    # Skill Score (si se proporciona baseline_mae)
    if baseline_mae is not None:
        # SS = 1 - (MAE_modelo / MAE_baseline)
        # SS = 1: modelo perfecto
        # SS = 0: modelo igual al baseline
        # SS < 0: modelo peor que el baseline
        # SS > 0: modelo mejor que el baseline
        skill_score = 1 - (mae / baseline_mae)
        result['SS'] = skill_score
    
    return result


def evaluate_model(
    y_true_dict: Dict[str, np.ndarray],
    y_pred_dict: Dict[str, np.ndarray],
    capacity: np.ndarray = None,
    baseline_mae_dict: Dict[str, float] = None
) -> pd.DataFrame:
    """
    Evalua el modelo para todos los horizontes.
    
    Parametros
    ----------
    y_true_dict : Dict[str, np.ndarray]
        Diccionario {target_name: valores_reales}
    y_pred_dict : Dict[str, np.ndarray]
        Diccionario {target_name: valores_predichos}
    capacity : np.ndarray, optional
        Capacidad de cada estacion para calcular MAE en bicicletas
    baseline_mae_dict : Dict[str, float], optional
        Diccionario {target_name: mae_baseline} para calcular Skill Score
        
    Retorna
    -------
    pd.DataFrame
        DataFrame con metricas por horizonte
    """
    results = []
    
    for target, horizon in zip(TARGETS, HORIZONS):
        baseline_mae = baseline_mae_dict.get(target) if baseline_mae_dict else None
        metrics = calculate_metrics(y_true_dict[target], y_pred_dict[target], capacity, baseline_mae)
        metrics['Horizonte'] = horizon
        results.append(metrics)
    
    df = pd.DataFrame(results)
    
    # Ordenar columnas
    cols = ['Horizonte', 'MAE', 'RMSE']
    if capacity is not None:
        cols.append('MAE_bikes')
    if baseline_mae_dict is not None:
        cols.append('SS')
    
    df = df[cols]
    
    return df


def save_metrics(metrics_df: pd.DataFrame, output_dir: str, filename: str = 'metrics.csv'):
    """
    Guarda las metricas en un archivo CSV.
    
    Parametros
    ----------
    metrics_df : pd.DataFrame
        DataFrame con las metricas
    output_dir : str
        Directorio de salida
    filename : str
        Nombre del archivo (default: 'metrics.csv')
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    filepath = path / filename
    metrics_df.to_csv(filepath, index=False)
    print(f"Metricas guardadas en: {filepath}")


def load_all_metrics(output_base: str = 'output') -> Dict[str, pd.DataFrame]:
    """
    Carga las metricas de todos los modelos.
    
    Parametros
    ----------
    output_base : str
        Directorio base donde estan las carpetas de modelos
        
    Retorna
    -------
    Dict[str, pd.DataFrame]
        Diccionario {nombre_modelo: DataFrame_metricas}
    """
    models = {
        'Persistencia': '0_persistence',
        'Regresion Lineal': '1_linear_regression',
        #'Reg. Lineal + Clusters': '1_linear_regression_cluster',
        'XGBoost': '2_xgboost',
        #'XGBoost + Clusters': '2_xgboost_cluster',
        'LSTM': '3_lstm'
    }
    
    metrics = {}
    for name, folder in models.items():
        path = Path(output_base) / folder / 'metrics.csv'
        if path.exists():
            metrics[name] = pd.read_csv(path)
            print(f"Cargadas metricas de {name}")
        else:
            print(f"No se encontraron metricas para {name} en {path}")
    
    return metrics


# =============================================================================
# Funciones de visualizacion
# =============================================================================

def setup_plot_style():
    """Configura el estilo de las graficas."""
    plt.style.use('seaborn-v0_8-notebook')


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int,
    output_dir: str,
    model_name: str
):
    """
    Grafica la distribucion de errores (residuos).
    
    Parametros
    ----------
    y_true : np.ndarray
        Valores reales
    y_pred : np.ndarray
        Valores predichos
    horizon : int
        Horizonte de prediccion en minutos
    output_dir : str
        Directorio de salida
    model_name : str
        Nombre del modelo para el titulo
    """
    setup_plot_style()
    
    errors = y_pred - y_true
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(errors, bins=100, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', label='Error = 0')
    ax.axvline(x=errors.mean(), color='green', linestyle='--', 
               label=f'Media = {errors.mean():.4f}')
    
    ax.set_xlabel('Error (prediccion - real)')
    ax.set_ylabel('Frecuencia')
    ax.set_title(f'Distribucion de errores - {model_name} (horizonte {horizon} min)')
    ax.legend()
    
    plt.tight_layout()
    
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / f'error_distribution_{horizon}min.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_residuals_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int,
    output_dir: str,
    model_name: str,
    sample_size: int = 10000
):
    """
    Grafica residuos vs valores predichos.
    
    Parametros
    ----------
    y_true : np.ndarray
        Valores reales
    y_pred : np.ndarray
        Valores predichos
    horizon : int
        Horizonte de prediccion en minutos
    output_dir : str
        Directorio de salida
    model_name : str
        Nombre del modelo
    sample_size : int
        Tamano de muestra para graficar (default: 10000)
    """
    setup_plot_style()
    
    residuals = y_pred - y_true
    
    # Muestrear si hay muchos puntos
    if len(residuals) > sample_size:
        idx = np.random.choice(len(residuals), sample_size, replace=False)
        residuals_sample = residuals[idx]
        y_pred_sample = y_pred[idx]
    else:
        residuals_sample = residuals
        y_pred_sample = y_pred
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(y_pred_sample, residuals_sample, alpha=0.3, s=1)
    ax.axhline(y=0, color='red', linestyle='--')
    
    ax.set_xlabel('Valor predicho')
    ax.set_ylabel('Residuo (prediccion - real)')
    ax.set_title(f'Residuos vs Prediccion - {model_name} (horizonte {horizon} min)')
    
    plt.tight_layout()
    
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / f'residuals_vs_predicted_{horizon}min.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_qq_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int,
    output_dir: str,
    model_name: str
):
    """
    Grafica QQ-plot de los residuos.
    
    Parametros
    ----------
    y_true : np.ndarray
        Valores reales
    y_pred : np.ndarray
        Valores predichos
    horizon : int
        Horizonte de prediccion en minutos
    output_dir : str
        Directorio de salida
    model_name : str
        Nombre del modelo
    """
    setup_plot_style()
    
    residuals = y_pred - y_true
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title(f'QQ-Plot de residuos - {model_name} (horizonte {horizon} min)')
    
    plt.tight_layout()
    
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / f'qq_plot_{horizon}min.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def select_stations_by_activity(
    df: pd.DataFrame,
    target: str
) -> Dict[str, str]:
    """
    Selecciona estaciones representativas segun su nivel de actividad.
    
    Parametros
    ----------
    df : pd.DataFrame
        DataFrame con los datos
    target : str
        Nombre de la columna target para calcular actividad
        
    Retorna
    -------
    Dict[str, str]
        Diccionario con estaciones: alta_actividad, baja_actividad, media_actividad
    """
    # Calcular varianza del target por estacion (proxy de actividad)
    station_activity = df.groupby('station_code')[target].var()
    
    # Estacion con mayor actividad (mayor varianza)
    high_activity = station_activity.idxmax()
    
    # Estacion con menor actividad (menor varianza, excluyendo ceros)
    non_zero_activity = station_activity[station_activity > 0]
    low_activity = non_zero_activity.idxmin()
    
    # Estacion con actividad media (mas cercana a la mediana)
    median_activity = station_activity.median()
    mid_activity = (station_activity - median_activity).abs().idxmin()
    
    return {
        'alta': high_activity,
        'baja': low_activity,
        'media': mid_activity
    } # type: ignore


def plot_prediction_example(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    target: str,
    horizon: int,
    output_dir: str,
    model_name: str,
    station_code: str,
    station_type: str,
    n_points: int = 144  # 1 dia con datos cada 10 min
):
    """
    Grafica un ejemplo de prediccion vs valores reales para una estacion.
    Genera dos graficas: una con delta de ocupacion y otra con bicicletas.
    
    Parametros
    ----------
    df : pd.DataFrame
        DataFrame con los datos (debe incluir snapshot_time, station_code, 
        target, capacity y ocu)
    y_pred : np.ndarray
        Predicciones del modelo (delta de ocupacion)
    target : str
        Nombre de la columna target
    horizon : int
        Horizonte de prediccion en minutos
    output_dir : str
        Directorio de salida
    model_name : str
        Nombre del modelo
    station_code : str
        Codigo de estacion
    station_type : str
        Tipo de estacion para el nombre del archivo (alta, baja, media)
    n_points : int
        Numero de puntos a graficar (default: 144 = 1 dia)
    """
    setup_plot_style()
    
    df_plot = df.copy()
    df_plot['prediction'] = y_pred
    
    # Filtrar por estacion
    df_station = df_plot[df_plot['station_code'] == station_code].copy()
    df_station = df_station.sort_values('snapshot_time').head(n_points)
    
    if len(df_station) == 0:
        print(f"    Advertencia: No hay datos para estacion {station_code}")
        return
    
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    # Mapeo de tipos a etiquetas en espanol
    type_labels = {
        'alta': 'alta actividad',
        'baja': 'baja actividad',
        'media': 'actividad media'
    }
    type_label = type_labels.get(station_type, station_type)
    
    # =========================================================================
    # Grafica: Bicicletas predichas vs reales
    # =========================================================================
    # Calcular bicicletas futuras reales y predichas
    # Bicicletas actuales = ocupacion actual * capacidad
    # Bicicletas futuras = capacidad * (ocupacion actual + delta de ocupacion)
    
    if 'ocu' in df_station.columns and 'capacity' in df_station.columns:
        # Bicicletas actuales (desde ocupacion)
        bikes_current = (df_station['ocu'] * df_station['capacity']).round().astype(int)
        
        # Bicicletas futuras reales (usando el delta real)
        bikes_future_real = (df_station['capacity'] * (df_station['ocu'] + df_station[target])).round().astype(int)
        
        # Bicicletas futuras predichas (usando el delta predicho)
        bikes_future_pred = (df_station['capacity'] * (df_station['ocu'] + df_station['prediction'])).round().astype(int)
        
        # Obtener capacidad
        capacity = int(df_station['capacity'].iloc[0])
        
        # Asegurar que no haya valores negativos ni mayores que la capacidad
        bikes_current = bikes_current.clip(lower=0, upper=capacity)
        bikes_future_real = bikes_future_real.clip(lower=0, upper=capacity)
        bikes_future_pred = bikes_future_pred.clip(lower=0, upper=capacity)
        
        # =====================================================================
        # Grafica 1: Lineas continuas
        # =====================================================================
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Graficar bicicletas actuales como referencia
        ax.plot(df_station['snapshot_time'], bikes_current, 
                label='Bicicletas actuales', alpha=0.5, color='gray', linewidth=1)
        
        # Graficar bicicletas futuras reales y predichas
        ax.plot(df_station['snapshot_time'], bikes_future_real, 
                label=f'Bicicletas reales (t+{horizon}min)', alpha=0.8)
        ax.plot(df_station['snapshot_time'], bikes_future_pred, 
                label=f'Bicicletas predichas (t+{horizon}min)', alpha=0.8, linestyle='--')
        
        # Mostrar capacidad como referencia
        ax.axhline(y=capacity, color='red', linestyle=':', alpha=0.5, 
                   label=f'Capacidad ({capacity})')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        ax.set_xlabel('Tiempo')
        ax.set_ylabel('Numero de bicicletas')
        ax.set_title(f'Prediccion de bicicletas - {model_name} - Estacion {station_code} ({type_label}, cap={capacity})')
        ax.legend()
        ax.set_ylim(bottom=-1, top=capacity + 2)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        fig.savefig(path / f'prediction_bikes_{horizon}min_{station_type}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # =====================================================================
        # Grafica 2: Step plot (escalera)
        # =====================================================================
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Graficar bicicletas actuales como referencia
        ax.step(df_station['snapshot_time'], bikes_current, 
                where='post', label='Bicicletas actuales', alpha=0.5, color='#222222', linewidth=1)
        
        # Graficar bicicletas futuras reales y predichas
        ax.step(df_station['snapshot_time'], bikes_future_real, 
                where='post', label=f'Bicicletas reales (t+{horizon}min)', alpha=0.8, linewidth=1)
        ax.step(df_station['snapshot_time'], bikes_future_pred, 
                where='post', label=f'Bicicletas predichas (t+{horizon}min)', alpha=0.8, linestyle='--', linewidth=1)
        
        # Mostrar capacidad como referencia
        ax.axhline(y=capacity, color='red', linestyle=':', alpha=0.5, 
                   label=f'Capacidad ({capacity})')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        ax.set_xlabel('Tiempo')
        ax.set_ylabel('Numero de bicicletas')
        ax.set_title(f'Prediccion de bicicletas (step) - {model_name} - Estacion {station_code} ({type_label}, cap={capacity})')
        ax.legend()
        ax.set_ylim(bottom=-1, top=capacity + 2)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        fig.savefig(path / f'prediction_bikes_step_{horizon}min_{station_type}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)


def plot_all_diagnostics(
    df_test: pd.DataFrame,
    y_true_dict: Dict[str, np.ndarray],
    y_pred_dict: Dict[str, np.ndarray],
    output_dir: str,
    model_name: str
):
    """
    Genera todas las graficas de diagnostico para un modelo.
    
    Parametros
    ----------
    df_test : pd.DataFrame
        DataFrame de test
    y_true_dict : Dict[str, np.ndarray]
        Diccionario con valores reales por target
    y_pred_dict : Dict[str, np.ndarray]
        Diccionario con predicciones por target
    output_dir : str
        Directorio de salida
    model_name : str
        Nombre del modelo
    """
    # Seleccionar estaciones representativas (usar primer target para calcular actividad)
    first_target = TARGETS[0]
    stations = select_stations_by_activity(df_test, first_target)
    print(f"  Estaciones seleccionadas:")
    print(f"    Alta actividad: {stations['alta']}")
    print(f"    Baja actividad: {stations['baja']}")
    print(f"    Actividad media: {stations['media']}")
    
    for target, horizon in zip(TARGETS, HORIZONS):
        y_true = y_true_dict[target]
        y_pred = y_pred_dict[target]
        
        print(f"  Generando graficas para horizonte {horizon} min...")
        
        # Distribucion de errores
        plot_error_distribution(y_true, y_pred, horizon, output_dir, model_name)
        
        # Residuos vs prediccion
        plot_residuals_vs_predicted(y_true, y_pred, horizon, output_dir, model_name)
        
        # QQ-plot
        plot_qq_residuals(y_true, y_pred, horizon, output_dir, model_name)
        
        # Ejemplos de prediccion por tipo de estacion
        for station_type, station_code in stations.items():
            plot_prediction_example(
                df_test, y_pred, target, horizon, output_dir, model_name,
                station_code=station_code,
                station_type=station_type
            )
    
    print(f"Graficas guardadas en: {output_dir}")


def plot_metrics_comparison(
    metrics_dict: Dict[str, pd.DataFrame],
    output_dir: str
):
    """
    Genera graficas comparativas de metricas entre modelos.
    
    Parametros
    ----------
    metrics_dict : Dict[str, pd.DataFrame]
        Diccionario {nombre_modelo: DataFrame_metricas}
    output_dir : str
        Directorio de salida
    """
    setup_plot_style()
    
    # Preparar datos
    models = list(metrics_dict.keys())
    horizons = HORIZONS
    
    mae_data = {model: metrics_dict[model]['MAE'].values for model in models}
    rmse_data = {model: metrics_dict[model]['RMSE'].values for model in models}
    da_data = {model: metrics_dict[model]['DA'].values for model in models}
    
    x = np.arange(len(horizons))
    width = 0.8 / len(models)
    
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    # Grafica MAE
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, model in enumerate(models):
        offset = (i - len(models)/2 + 0.5) * width
        ax.bar(x + offset, mae_data[model], width, label=model)
    
    ax.set_xlabel('Horizonte de prediccion (minutos)')
    ax.set_ylabel('MAE')
    ax.set_title('Comparacion de MAE por modelo y horizonte')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{h} min' for h in horizons])
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(path / 'comparison_mae.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Grafica RMSE
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, model in enumerate(models):
        offset = (i - len(models)/2 + 0.5) * width
        ax.bar(x + offset, rmse_data[model], width, label=model)
    
    ax.set_xlabel('Horizonte de prediccion (minutos)')
    ax.set_ylabel('RMSE')
    ax.set_title('Comparacion de RMSE por modelo y horizonte')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{h} min' for h in horizons])
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(path / 'comparison_rmse.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Grafica DA (Directional Accuracy)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, model in enumerate(models):
        offset = (i - len(models)/2 + 0.5) * width
        ax.bar(x + offset, da_data[model], width, label=model)
    
    ax.set_xlabel('Horizonte de prediccion (minutos)')
    ax.set_ylabel('Directional Accuracy (%)')
    ax.set_title('Comparacion de Directional Accuracy por modelo y horizonte')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{h} min' for h in horizons])
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Aleatorio (50%)')
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(path / 'comparison_da.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Graficas de comparacion guardadas en: {output_dir}")


def create_comparison_table(metrics_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Crea una tabla resumen comparativa de todos los modelos.
    
    Parametros
    ----------
    metrics_dict : Dict[str, pd.DataFrame]
        Diccionario {nombre_modelo: DataFrame_metricas}
        
    Retorna
    -------
    pd.DataFrame
        Tabla resumen con MAE, RMSE, SS y MAE_bikes por modelo y horizonte
    """
    rows = []
    
    for model_name, metrics_df in metrics_dict.items():
        for _, row in metrics_df.iterrows():
            row_data = {
                'Modelo': model_name,
                'Horizonte': int(row['Horizonte']),
                'MAE': row['MAE'],
                'RMSE': row['RMSE']
            }
            
            # Agregar metricas opcionales si existen
            if 'SS' in row:
                row_data['SS'] = row['SS']
            if 'MAE_bikes' in row:
                row_data['MAE_bikes'] = row['MAE_bikes']
            
            rows.append(row_data)
    
    df = pd.DataFrame(rows)
    
    # Determinar valores disponibles
    available_values = ['MAE', 'RMSE']
    if 'SS' in df.columns:
        available_values.append('SS')
    if 'MAE_bikes' in df.columns:
        available_values.append('MAE_bikes')
    
    # Crear tabla pivote
    table = df.pivot(index='Modelo', columns='Horizonte', values=available_values)
    
    # Ordenar modelos
    model_order = ['Persistencia', 'Regresion Lineal', 'XGBoost', 'LSTM']
    existing_models = [m for m in model_order if m in table.index]
    table = table.loc[existing_models]
    
    return table
