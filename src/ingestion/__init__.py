"""
Módulo de Ingestão de Dados de Satélite (Sentinel-2 L2A).
"""

from .sentinel_reader import (
    SentinelScene,
    read_sentinel_bands,
    calculate_ndwi,
    build_glint_mask,
    build_clean_water_mask,
    apply_water_mask,
    save_geotiff,
)

__all__ = [
    "SentinelScene",
    "read_sentinel_bands",
    "calculate_ndwi",
    "build_glint_mask",
    "build_clean_water_mask",
    "apply_water_mask",
    "save_geotiff",
]
