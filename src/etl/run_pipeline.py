"""
Pipeline ETL completo para procesamiento de datos de Ecobici.

Este script ejecuta el pipeline completo de ETL en el orden correcto para generar
el dataset final de entrenamiento en data/ecobici/gbfs/2_train/2_gbfs_train.parquet

Flujo del pipeline:
1. ESTACIONES: Extraccion y enriquecimiento de estaciones activas
2. VIAJES: Limpieza, enriquecimiento y calculo de caracteristicas de actividad
3. GBFS: Limpieza, remuestreo, enriquecimiento y finalizacion del dataset de entrenamiento

Dependencias externas (deben existir previamente):
- data/holidays/holidays.csv
- data/pois/*.geojson
- data/transit/*.geojson
- data/ids/ids.geojson
- data/meteo/station/1km/*.parquet (opcional)
- Datos crudos de GBFS (station_information.json, station_status.json)
- Datos crudos de viajes (CSV)

Uso:
    python run_pipeline.py [--skip-stations] [--skip-trips] [--skip-gbfs]
    
Opciones:
    --skip-stations  Omite el procesamiento de estaciones
    --skip-trips     Omite el procesamiento de viajes
    --skip-gbfs      Omite el procesamiento de GBFS
    --step STEP      Ejecuta solo un paso especifico (stations, trips, gbfs)
"""

import argparse
import sys
import subprocess
from pathlib import Path
from pyprojroot import here
from datetime import datetime

BASE_DIR = here()
ETL_DIR = BASE_DIR / "src/etl"


class PipelineStep:
    """Representa un paso del pipeline ETL."""
    
    def __init__(self, name: str, script_path: Path, output_files: list[Path], description: str):
        self.name = name
        self.script_path = script_path
        self.output_files = output_files
        self.description = description
        
    def run(self) -> bool:
        """Ejecuta el script del paso."""
        print(f"\n{'='*80}")
        print(f"EJECUTANDO: {self.name}")
        print(f"Script: {self.script_path.relative_to(BASE_DIR)}")
        print(f"Descripcion: {self.description}")
        print(f"{'='*80}\n")
        
        try:
            result = subprocess.run(
                ["uv", "run", str(self.script_path)],
                cwd=BASE_DIR,
                check=True,
                capture_output=False,
                text=True
            )
            
            print(f"\n✓ Completado: {self.name}")
            
            missing_files = [f for f in self.output_files if not f.exists()]
            if missing_files:
                print(f"\nADVERTENCIA: Algunos archivos de salida no fueron encontrados:")
                for f in missing_files:
                    print(f"  - {f.relative_to(BASE_DIR)}")
                return False
            else:
                print(f"Archivos de salida verificados:")
                for f in self.output_files:
                    print(f"  ✓ {f.relative_to(BASE_DIR)}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"\n✗ ERROR en {self.name}")
            print(f"Codigo de salida: {e.returncode}")
            return False
        except Exception as e:
            print(f"\n✗ ERROR INESPERADO en {self.name}: {e}")
            return False
    
    def check_outputs_exist(self) -> bool:
        """Verifica si los archivos de salida ya existen."""
        return all(f.exists() for f in self.output_files)


class ETLPipeline:
    """Pipeline completo de ETL para datos de Ecobici."""
    
    def __init__(self):
        self.data_dir = BASE_DIR / "data"
        self.steps = self._define_steps()
        self.start_time = None
        self.end_time = None
        
    def _define_steps(self) -> dict:
        """Define todos los pasos del pipeline organizados por categoria."""
        
        stations_dir = self.data_dir / "ecobici/stations"
        trips_clean_dir = self.data_dir / "ecobici/trips/1_clean"
        trips_enriched_dir = self.data_dir / "ecobici/trips/2_enriched"
        gbfs_clean_dir = self.data_dir / "ecobici/gbfs/1_clean"
        gbfs_train_dir = self.data_dir / "ecobici/gbfs/2_train"
        
        return {
            "stations": [
                PipelineStep(
                    name="Extraccion de Estaciones Activas",
                    script_path=ETL_DIR / "stations/0_extract_active_stations.py",
                    output_files=[stations_dir / "0_stations.csv"],
                    description="Extrae estaciones activas de archivos station_information.json"
                ),
                PipelineStep(
                    name="Enriquecimiento de Estaciones",
                    script_path=ETL_DIR / "stations/1_enrich.py",
                    output_files=[stations_dir / "1_stations_enriched.parquet"],
                    description="Enriquece estaciones con POIs, transito, IDS, UTM y clusters"
                ),
            ],
            "trips": [
                PipelineStep(
                    name="Limpieza de Viajes",
                    script_path=ETL_DIR / "trips/0_clean.py",
                    output_files=[trips_clean_dir],
                    description="Convierte CSV de viajes a Parquet particionado y limpia datos"
                ),
                PipelineStep(
                    name="Enriquecimiento de Viajes",
                    script_path=ETL_DIR / "trips/1_enrich.py",
                    output_files=[trips_enriched_dir],
                    description="Enriquece viajes con informacion de dias festivos"
                ),
                PipelineStep(
                    name="Caracteristicas de Actividad de Estaciones",
                    script_path=ETL_DIR / "stations/2_trips_features.py",
                    output_files=[stations_dir / "2_stations_activity_features.parquet"],
                    description="Calcula flujo neto e intensidad de uso por estacion"
                ),
            ],
            "gbfs": [
                PipelineStep(
                    name="Limpieza de GBFS",
                    script_path=ETL_DIR / "gbfs/0_clean.py",
                    output_files=[gbfs_clean_dir],
                    description="Convierte JSON de GBFS a Parquet particionado"
                ),
                PipelineStep(
                    name="Remuestreo de GBFS",
                    script_path=ETL_DIR / "gbfs/1_gbfs_resample.py",
                    output_files=[gbfs_train_dir / "0_gbfs_resampled.parquet"],
                    description="Remuestrea datos GBFS a intervalos de 10 minutos"
                ),
                PipelineStep(
                    name="Enriquecimiento de GBFS",
                    script_path=ETL_DIR / "gbfs/2_gbfs_enrich.py",
                    output_files=[gbfs_train_dir / "1_gbfs_train.parquet"],
                    description="Enriquece GBFS con tiempo ciclico, estaciones, actividad, meteorologia y festivos"
                ),
                PipelineStep(
                    name="Finalizacion de Datos de Entrenamiento",
                    script_path=ETL_DIR / "gbfs/3_train_data.py",
                    output_files=[gbfs_train_dir / "2_gbfs_train.parquet"],
                    description="Limpieza final y calculo de ocupacion para dataset de entrenamiento"
                ),
            ],
        }
    
    def run(self, skip_categories: set[str] = None, only_step: str = None):
        """
        Ejecuta el pipeline completo o pasos especificos.
        
        Args:
            skip_categories: Conjunto de categorias a omitir ('stations', 'trips', 'gbfs')
            only_step: Si se especifica, ejecuta solo esa categoria
        """
        skip_categories = skip_categories or set()
        
        self.start_time = datetime.now()
        print("\n" + "="*80)
        print("INICIANDO PIPELINE ETL DE ECOBICI")
        print(f"Hora de inicio: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        if only_step:
            if only_step not in self.steps:
                print(f"\n✗ ERROR: Paso '{only_step}' no existe.")
                print(f"Pasos disponibles: {', '.join(self.steps.keys())}")
                return False
            categories_to_run = [only_step]
        else:
            categories_to_run = [cat for cat in self.steps.keys() if cat not in skip_categories]
        
        success_count = 0
        failure_count = 0
        skipped_count = 0
        
        for category in categories_to_run:
            print(f"\n{'#'*80}")
            print(f"# CATEGORIA: {category.upper()}")
            print(f"{'#'*80}")
            
            for step in self.steps[category]:
                if step.check_outputs_exist():
                    print(f"\n⊙ OMITIENDO: {step.name}")
                    print(f"  Los archivos de salida ya existen:")
                    for f in step.output_files:
                        if f.exists():
                            print(f"    ✓ {f.relative_to(BASE_DIR)}")
                    print(f"  Para regenerar, elimina los archivos de salida primero.")
                    skipped_count += 1
                    continue
                
                if step.run():
                    success_count += 1
                else:
                    failure_count += 1
                    print(f"\n¿Deseas continuar con el siguiente paso? (s/N): ", end="")
                    response = input().strip().lower()
                    if response not in ['s', 'si', 'y', 'yes']:
                        print("\nPipeline detenido por el usuario.")
                        self._print_summary(success_count, failure_count, skipped_count)
                        return False
        
        self.end_time = datetime.now()
        self._print_summary(success_count, failure_count, skipped_count)
        
        return failure_count == 0
    
    def _print_summary(self, success: int, failure: int, skipped: int):
        """Imprime resumen de ejecucion del pipeline."""
        duration = (self.end_time - self.start_time) if self.end_time else None
        
        print("\n" + "="*80)
        print("RESUMEN DE EJECUCION DEL PIPELINE")
        print("="*80)
        print(f"Pasos exitosos:  {success}")
        print(f"Pasos fallidos:   {failure}")
        print(f"Pasos omitidos:   {skipped}")
        
        if duration:
            print(f"\nHora de inicio:   {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Hora de fin:      {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Duracion total:   {duration}")
        
        if failure == 0 and success > 0:
            final_output = self.data_dir / "ecobici/gbfs/2_train/2_gbfs_train.parquet"
            if final_output.exists():
                print(f"\n✓ PIPELINE COMPLETADO EXITOSAMENTE")
                print(f"  Dataset final disponible en:")
                print(f"  {final_output.relative_to(BASE_DIR)}")
            else:
                print(f"\n⊙ Pipeline ejecutado pero el archivo final no esta disponible")
        elif failure > 0:
            print(f"\n✗ PIPELINE COMPLETADO CON ERRORES")
        
        print("="*80 + "\n")
    
    def check_dependencies(self) -> tuple[list[Path], list[Path]]:
        """
        Verifica la existencia de dependencias externas requeridas.
        
        Returns:
            Tupla de (archivos_existentes, archivos_faltantes)
        """
        required_files = [
            self.data_dir / "holidays/holidays.csv",
        ]
        
        optional_dirs = [
            self.data_dir / "pois",
            self.data_dir / "transit",
            self.data_dir / "ids",
            self.data_dir / "meteo/station/1km",
        ]
        
        existing = []
        missing = []
        
        for file in required_files:
            if file.exists():
                existing.append(file)
            else:
                missing.append(file)
        
        for dir_path in optional_dirs:
            if dir_path.exists():
                existing.append(dir_path)
            else:
                missing.append(dir_path)
        
        return existing, missing


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline ETL completo para procesamiento de datos de Ecobici",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python run_pipeline.py                    # Ejecuta pipeline completo
  python run_pipeline.py --skip-stations    # Omite procesamiento de estaciones
  python run_pipeline.py --step gbfs        # Ejecuta solo paso de GBFS
  python run_pipeline.py --check-deps       # Verifica dependencias
        """
    )
    
    parser.add_argument(
        "--skip-stations",
        action="store_true",
        help="Omite el procesamiento de estaciones"
    )
    
    parser.add_argument(
        "--skip-trips",
        action="store_true",
        help="Omite el procesamiento de viajes"
    )
    
    parser.add_argument(
        "--skip-gbfs",
        action="store_true",
        help="Omite el procesamiento de GBFS"
    )
    
    parser.add_argument(
        "--step",
        choices=["stations", "trips", "gbfs"],
        help="Ejecuta solo un paso especifico"
    )
    
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Verifica dependencias y sale"
    )
    
    args = parser.parse_args()
    
    pipeline = ETLPipeline()
    
    if args.check_deps:
        print("\n" + "="*80)
        print("VERIFICACION DE DEPENDENCIAS")
        print("="*80 + "\n")
        
        existing, missing = pipeline.check_dependencies()
        
        if existing:
            print("Dependencias encontradas:")
            for dep in existing:
                print(f"  ✓ {dep.relative_to(BASE_DIR)}")
        
        if missing:
            print("\nDependencias faltantes (opcionales):")
            for dep in missing:
                print(f"  ✗ {dep.relative_to(BASE_DIR)}")
        
        print("\n" + "="*80 + "\n")
        return
    
    skip_categories = set()
    if args.skip_stations:
        skip_categories.add("stations")
    if args.skip_trips:
        skip_categories.add("trips")
    if args.skip_gbfs:
        skip_categories.add("gbfs")
    
    success = pipeline.run(skip_categories=skip_categories, only_step=args.step)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
