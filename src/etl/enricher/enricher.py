"""
Protocolos para enriquecedores de datos.
Define las interfaces que deben implementar los enriquecedores para estaciones,
viajes y datos GBFS.
"""
import polars as pl
from typing import Protocol


class StationEnricher(Protocol):
    """
    Protocolo para clases de enriquecimiento de estaciones.
    Cualquier clase que implemente este protocolo debe proporcionar un método enrich
    que tome un DataFrame de estaciones y devuelva un DataFrame enriquecido.
    """

    def enrich(self, df: pl.DataFrame) -> pl.DataFrame: ...


class TripEnricher(Protocol):
    """
    Protocolo para clases de enriquecimiento de viajes.
    Cualquier clase que implemente este protocolo debe proporcionar un método enrich
    que tome un DataFrame de viajes y devuelva un DataFrame enriquecido.
    """

    def enrich(self, df: pl.DataFrame) -> pl.DataFrame: ...


class GBFSEnricher(Protocol):
    """
    Protocolo para clases de enriquecimiento de GBFS.
    Cualquier clase que implemente este protocolo debe proporcionar un método enrich
    que tome un DataFrame de GBFS y devuelva un DataFrame enriquecido.
    """

    def enrich(self, df: pl.DataFrame) -> pl.DataFrame: ...
