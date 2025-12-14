"""
Enriquecedor que agrega características de rezago (lags) y columnas objetivo a los datos GBFS.
"""
import polars as pl


class LagsEnricher:
    """
    Enriquecedor que agrega caracteristicas de rezago (lags), tendencias y columnas objetivo
    para datos GBFS.
    
    Crea:
    - Caracteristica de ocupacion (occupancy = bikes / capacity)
    
    - Versiones rezagadas de bikes para capturar patrones temporales:
      * Rezagos inmediatos (t-1, t-2): Inercia actual (10-20 min)
      * Rezagos recientes (t-6, t-12): Tendencia reciente (1-2 horas)
      * Rezagos estacionales (t-138, t-144): Mismo momento del dia anterior (23-24 horas)
    
    - Caracteristicas de tendencia para cada lag:
      * bikes_trend_{lag}: Cambio en bicicletas desde el lag (bikes_t - bikes_t-lag)
      * Valores positivos indican aumento, negativos indican disminucion
      * Captura la direccion del cambio en diferentes escalas temporales
    
    - Columnas objetivo (valores futuros de bikes) para entrenamiento de modelos:
      * target_bikes_20min: Bikes 20 minutos adelante (t+2)
      * target_bikes_40min: Bikes 40 minutos adelante (t+4)
      * target_bikes_60min: Bikes 60 minutos adelante (t+6)
    
    Todas las caracteristicas se calculan por estacion para evitar mezclar datos entre estaciones.
    
    Lags configurados por defecto:
    - Inmediatos: 1, 2 (10, 20 minutos)
    - Recientes: 6, 12 (60, 120 minutos)
    - Estacionales: 138, 144 (1380, 1440 minutos = 23, 24 horas)
    """
    
    def __init__(
        self,
        immediate_lags: list[int] = [1, 2],
        recent_lags: list[int] = [6, 12],
        seasonal_lags: list[int] = [138, 144],  # 1 dia = 144 pasos de 10 min (24*6)
        create_targets: bool = True,
        target_horizons: dict[str, int] = None, # type: ignore
    ):
        """
        Inicializa el enriquecedor de rezagos combinado para datos de 10 minutos.
        
        Args:
            immediate_lags: Rezagos a corto plazo (default: [1, 2] = 10, 20 min)
            recent_lags: Rezagos a mediano plazo (default: [6, 12] = 60, 120 min)
            seasonal_lags: Rezagos estacionales a largo plazo (default: [138, 144] = 23, 24 horas)
            create_targets: Si se deben crear columnas objetivo
            target_horizons: Diccionario mapeando nombres de objetivos a desplazamientos 
                           (default: 20min=-2, 40min=-4, 60min=-6)
        """
        self.immediate_lags = immediate_lags
        self.recent_lags = recent_lags
        self.seasonal_lags = seasonal_lags
        self.create_targets = create_targets
        
        # Horizontes objetivo por defecto para intervalos de 10 minutos (desplazamientos negativos = mirar hacia adelante)
        if target_horizons is None:
            self.target_horizons = {
                "target_bikes_20min": -2,   # 20 min ahead (2 * 10 min)
                "target_bikes_40min": -4,   # 40 min ahead (4 * 10 min)
                "target_bikes_60min": -6,   # 60 min ahead (6 * 10 min)
            }
        else:
            self.target_horizons = target_horizons
        
        # Combinar todos los rezagos
        self.all_lags = sorted(immediate_lags + recent_lags + seasonal_lags)
    
    def enrich(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Agrega caracteristicas de ocupacion, rezagos de bikes, tendencias y columnas objetivo
        al dataframe.
        
        Proceso:
        1. Calcula ocupacion (bikes / capacity)
        2. Crea rezagos de bikes para cada lag configurado
        3. Calcula tendencias (cambio desde cada lag)
        4. Crea columnas objetivo (valores futuros)
        
        Los rezagos, tendencias y objetivos se calculan por estacion usando funciones de ventana
        para evitar mezclar datos entre diferentes estaciones.
        
        Args:
            df: DataFrame con columnas 'bikes', 'capacity', 'station_code', 'snapshot_time'
        
        Returns:
            DataFrame enriquecido con:
            - occupancy: Ocupacion actual
            - bikes_lag_{n}: Bicicletas en t-n
            - bikes_trend_{n}: Cambio desde t-n (bikes_t - bikes_t-n)
            - target_bikes_{horizon}: Bicicletas en t+horizon (si create_targets=True)
        """
        print(f"  Creando caracteristica de ocupacion (occupancy = bikes / capacity)...")
        
        # Calcular ocupacion
        df = df.with_columns([
            (pl.col("bikes") / pl.col("capacity")).alias("occupancy")
        ])
        
        # Asegurar que datos estan ordenados por estacion y tiempo
        df = df.sort(["station_code", "snapshot_time"])
        
        # === REZAGOS DE bikes ===
        print(f"\n  Creando caracteristicas de rezago para 'bikes' (paso 10 min)...")
        print(f"    Rezagos inmediatos: {self.immediate_lags}")
        print(f"    Rezagos recientes: {self.recent_lags}")
        print(f"    Rezagos estacionales: {self.seasonal_lags}")
        
        # Crear caracteristicas de rezago agrupadas por estacion para bikes
        bikes_lag_exprs = []
        for lag in self.all_lags:
            bikes_lag_exprs.append(
                pl.col("bikes")
                .shift(lag)
                .over("station_code")
                .alias(f"bikes_lag_{lag}")
            )
        
        # Aplicar transformaciones de rezago para bikes
        df = df.with_columns(bikes_lag_exprs)
        
        # === TENDENCIAS (TRENDS) PARA CADA LAG ===
        print(f"\n  Calculando tendencias para cada lag...")
        trend_exprs = []
        for lag in self.all_lags:
            # Tendencia = bikes actual - bikes en el lag
            # Valores positivos: aumento de bicicletas desde el lag
            # Valores negativos: disminucion de bicicletas desde el lag
            trend_exprs.append(
                (pl.col("bikes") - pl.col(f"bikes_lag_{lag}"))
                .alias(f"bikes_trend_{lag}")
            )
        
        # Aplicar transformaciones de tendencia
        df = df.with_columns(trend_exprs)
        print(f"    Creadas {len(self.all_lags)} caracteristicas de tendencia")
        
        max_lag = max(self.all_lags)
        total_rows = df.shape[0]
        
        # Contar nulls en el rezago mas grande
        null_count = df.filter(
            pl.col(f"bikes_lag_{max_lag}").is_null()
        ).shape[0]
        
        print(f"    Creadas {len(self.all_lags)} caracteristicas de rezago para bikes")
        print(f"    Filas con nulos en el rezago mas grande (t-{max_lag}): {null_count:,} ({null_count/total_rows*100:.2f}%)")
        
        # === COLUMNAS OBJETIVO (bikes) ===
        if self.create_targets:
            print(f"\n  Creando columnas objetivo para 'bikes' (paso 10 min)...")
            target_exprs = []
            for target_name, shift in self.target_horizons.items():
                # Target de bikes
                target_exprs.append(
                    pl.col("bikes")
                    .shift(shift)  # Negative shift = looking forward
                    .over("station_code")
                    .alias(target_name)
                )
                
                minutes = abs(shift) * 10  # Convertir pasos a minutos (intervalos de 10 min)
                print(f"    {target_name}: {minutes} minutos adelante (shift={shift})")
            
            # Aplicar transformaciones objetivo
            df = df.with_columns(target_exprs)
            
            # Reportar valores null en objetivos (ultimas filas por estacion tendran nulls)
            target_col_name = list(self.target_horizons.keys())[0]
            target_null_count = df.filter(pl.col(target_col_name).is_null()).shape[0]
            
            print(f"    Creadas {len(self.target_horizons)} columnas objetivo de bikes")
            print(f"    Filas con nulos en objetivos bikes: {target_null_count:,} ({target_null_count/total_rows*100:.2f}%)")
        
        return df
