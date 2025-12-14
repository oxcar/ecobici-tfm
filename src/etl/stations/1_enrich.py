"""
Enriquecimiento de datos de estaciones con información adicional como POIs, tránsito, IDS, UTM y clusters.
"""
import os
import sys
from pathlib import Path
from typing import List

import polars as pl
from pyprojroot import here

# Agregar directorio padre al path para encontrar modulo enricher
sys.path.insert(0, str(here() / "src/etl"))

from enricher.enricher import StationEnricher
from enricher.stations.poi_enricher import POIEnricher
from enricher.stations.ids_enricher import IDSEnricher
from enricher.stations.transit_enricher import TransitEnricher
from enricher.stations.utm_enricher import UTMEnricher
from enricher.stations.cluster_enricher import ClusterEnricher

BASE_DIR = here()
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
POIS_DIR = DATA_DIR / "pois"
TRANSIT_DIR = DATA_DIR / "transit"
IDS_FILE = DATA_DIR / "ids/ids.geojson"
CLUSTERS_FILE = DATA_DIR / "ecobici/stations/3_station_clusters.csv"


def process_stations(
    stations_path: str | Path, output_dir: str | Path, enrichers: List[StationEnricher]
):
    input_path = Path(stations_path)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"El archivo de estaciones {input_path} no existe.")
        return

    print(f"Procesando estaciones desde {input_path}...")

    try:
        df = pl.read_csv(input_path, schema_overrides={"station_code": pl.String})

        # Columnas iniciales a eliminar
        initial_drop_cols = [
            "external_id",
            "rental_methods",
            "electric_bike_surcharge_waiver",
            "is_charging",
            "eightd_has_key_dispenser",
            "has_kiosk",
        ]
        df = df.drop([c for c in initial_drop_cols if c in df.columns])

        for enricher in enrichers:
            df = enricher.enrich(df)

        # Consolidar todas las columnas a eliminar despues del enriquecimiento
        final_drop_cols = {
            "station_id_right",
            "station_id_right_right",
            "peak_morning",
            "peak_evening",
            "peak_morning_evening",
            "peak_none",
            "departures",
            "arrivals",
        }

        # Identificar columnas por patron (usar set para mejor performance)
        cols_to_drop = [
            c for c in df.columns
            if c in final_drop_cols or c.endswith(("_500m", "_mean_dist_3_m"))
        ]

        if cols_to_drop:
            df = df.drop(cols_to_drop)

        output_path.mkdir(parents=True, exist_ok=True)
        dest_path = output_path / f"1_stations_enriched.parquet"
        df.write_parquet(dest_path)
        print(f"  - Guardado en {dest_path}")

    except Exception as e:
        print(f"Error procesando estaciones: {e}")


if __name__ == "__main__":

    stations_file = DATA_DIR / "ecobici/stations/0_stations.csv"
    output_dir = DATA_DIR / "ecobici/stations/"

    print(f"Archivo de estaciones: {stations_file}")
    print(f"Directorio de salida: {output_dir}")

    enrichers = []

    # Agregar enriquecedores de POI
    if POIS_DIR.exists():
        for poi_file in POIS_DIR.glob("*.geojson"):
            # Extraer categoria del nombre de archivo (ej. poi_food.geojson -> food)
            category = poi_file.stem.replace("poi_", "")
            print(f"Agregando enriquecedor de POI para {category} ({poi_file.name})")
            enrichers.append(POIEnricher(poi_file, category))
    else:
        print(f"Directorio de POIs {POIS_DIR} no encontrado.")

    # Agregar enriquecedor de Transito (fusionado todas las categorias)
    if TRANSIT_DIR.exists():
        transit_files = list(TRANSIT_DIR.glob("*.geojson"))
        if transit_files:
            print(f"Agregando enriquecedor de Tránsito fusionado con {len(transit_files)} categorías:")
            for tf in transit_files:
                print(f"  - {tf.name}")
            enrichers.append(TransitEnricher(transit_files)) # type: ignore
        else:
            print(f"No se encontraron archivos GeoJSON de tránsito en {TRANSIT_DIR}")
    else:
        print(f"Directorio de tránsito {TRANSIT_DIR} no encontrado.")

    # Agregar enriquecedor de IDS
    if IDS_FILE.exists():
        print(f"Agregando enriquecedor de IDS ({IDS_FILE.name})")
        enrichers.append(IDSEnricher(IDS_FILE))
    else:
        print(f"Archivo IDS {IDS_FILE} no encontrado.")

    # Agregar enriquecedor de UTM (siempre habilitado - convierte lat/lon a UTM 14N)
    print("Agregando enriquecedor de coordenadas UTM (WGS84 → UTM 14N)")
    enrichers.append(UTMEnricher(utm_zone="32614"))

    # Agregar enriquecedor de Clusters
    if CLUSTERS_FILE.exists():
        print(f"Agregando enriquecedor de Clusters ({CLUSTERS_FILE.name})")
        enrichers.append(ClusterEnricher(CLUSTERS_FILE))
    else:
        print(f"Archivo de clusters {CLUSTERS_FILE} no encontrado.")

    process_stations(stations_file, output_dir, enrichers)
