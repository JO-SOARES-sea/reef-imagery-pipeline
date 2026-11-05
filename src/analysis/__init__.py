"""
Módulo de Análise e Extração de Assinaturas de Substrato Marinho.

Os símbolos públicos são carregados sob demanda (PEP 562): importar ``src.analysis.features``
não deve puxar geopandas/sklearn (reef_segmenter) nem PIL (gemini_analyzer).
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "compute_bpi": ".features",
    "compute_focal_std": ".features",
    "compute_psdb": ".features",
    "compute_slope": ".features",
    "compute_vrm": ".features",
    "extract_morpho_spectral_features": ".features",
    "resample_emodnet_screen": ".features",
    "create_rgb_patch": ".gemini_analyzer",
    "describe_patch_with_gemini": ".gemini_analyzer",
    "ask_reef_noul": ".gemini_analyzer",
    "interpret_reef_noul": ".gemini_analyzer",
    "route_and_validate_with_jev": ".gemini_analyzer",
    "compute_reef_probability_map": ".reef_segmenter",
    "extract_candidate_reefs": ".reef_segmenter",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
