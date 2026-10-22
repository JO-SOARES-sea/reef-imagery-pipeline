"""
Módulo de Análise e Extração de Assinaturas de Substrato Marinho.
"""

from .features import (
    compute_psdb,
    compute_slope,
    compute_bpi,
    compute_vrm,
    compute_focal_std,
    resample_emodnet_screen,
    extract_morpho_spectral_features,
)
from .reef_segmenter import (
    compute_reef_probability_map,
    extract_candidate_reefs,
)
from .gemini_analyzer import (
    create_rgb_patch,
    describe_patch_with_gemini,
    interpret_reef_noul,
    route_and_validate_with_jev,
)

__all__ = [
    "compute_psdb",
    "compute_slope",
    "compute_bpi",
    "compute_vrm",
    "compute_focal_std",
    "resample_emodnet_screen",
    "extract_morpho_spectral_features",
    "compute_reef_probability_map",
    "extract_candidate_reefs",
    "create_rgb_patch",
    "describe_patch_with_gemini",
    "interpret_reef_noul",
    "route_and_validate_with_jev",
]
