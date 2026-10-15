"""
src/band_registry.py — Which genuine Sentinel-2 bands and preprocessing steps each feature needs.

Missing-band policy
-------------------
* A feature is computed only when every required band is a genuine observation of the same product,
  with known radiometry and at least some valid pixels.
* A missing/invalid band is NEVER replaced: no zeros, no copies of another band, no scaled bands
  (e.g. the unverified ``B04 = 0.8*B02`` / ``B08 = B02`` substitutions), no silent NaN fed to a model.
* If a feature that gates the rest of the pipeline is unavailable, the pipeline returns
  ``status="unavailable"`` — not a probability of zero and not a result from a model trained on a
  different input schema.
* Reduced-band models are out of scope until one is separately trained and spatially validated.

Each entry lists: required bands, required upstream features, preprocessing, and whether the
feature is *gating* (its absence makes the detector's main output unavailable).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

# Canonical band ids used by the reef detector (Earth Search asset keys in brackets).
BAND_ALIASES = {"B02": "blue", "B03": "green", "B04": "red", "B08": "nir"}


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    bands: tuple[str, ...]
    depends_on: tuple[str, ...] = ()
    preprocessing: tuple[str, ...] = ()
    gating: bool = False
    note: str = ""


_COMMON_PRE = ("same product/grid", "nodata masked before arithmetic", "DN→reflectance once from explicit metadata")

REEF_DETECTOR_FEATURES: dict[str, FeatureSpec] = {
    f.name: f
    for f in (
        FeatureSpec(
            "water_mask", ("B02", "B03", "B08"), (), _COMMON_PRE + ("NIR < GLINT_NIR_MAX (land + sun-glint rejection)",),
            gating=True,
            note="NIR is the only land/glint discriminator in this detector; no NIR → no mask → no detector output.",
        ),
        FeatureSpec(
            "deep_water_background", ("B02", "B03"), ("water_mask",), _COMMON_PRE + ("1st percentile over clean water",),
            gating=True,
        ),
        FeatureSpec(
            "lyzenga_dii", ("B02", "B03"), ("water_mask", "deep_water_background"),
            _COMMON_PRE + ("log(R - Lsw)", "auto k-ratio"), gating=True,
        ),
        FeatureSpec(
            "reef_likelihood", ("B03",), ("lyzenga_dii",), _COMMON_PRE + ("local dark-anomaly z", "3-px texture"),
            gating=True,
        ),
        FeatureSpec("clarity_nir_glint", ("B08",), (), _COMMON_PRE),
        FeatureSpec("clarity_red", ("B04",), (), _COMMON_PRE),
        FeatureSpec("clarity_blue_green_contrast", ("B02", "B03"), (), _COMMON_PRE),
        FeatureSpec("true_colour_overlay", ("B02", "B03", "B04"), (), _COMMON_PRE, note="display only"),
    )
}


@dataclass
class Availability:
    available: dict[str, bool] = field(default_factory=dict)
    missing_bands: dict[str, list[str]] = field(default_factory=dict)
    blocked_by: dict[str, list[str]] = field(default_factory=dict)

    def ok(self, feature: str) -> bool:
        return self.available.get(feature, False)

    def gating_failures(self, registry: Mapping[str, FeatureSpec] = REEF_DETECTOR_FEATURES) -> list[str]:
        return [n for n, s in registry.items() if s.gating and not self.available.get(n, False)]

    def as_dict(self) -> dict:
        return {"available": dict(self.available), "missing_bands": dict(self.missing_bands),
                "blocked_by": dict(self.blocked_by)}


def evaluate(usable_bands: Iterable[str],
             registry: Mapping[str, FeatureSpec] = REEF_DETECTOR_FEATURES) -> Availability:
    """
    Resolve which features can be computed from the set of *usable* bands (genuine, radiometry known,
    with valid pixels). Dependencies are resolved transitively in registry order.
    """
    have = set(usable_bands)
    av = Availability()
    for name, spec in registry.items():
        miss = [b for b in spec.bands if b not in have]
        deps = [d for d in spec.depends_on if not av.available.get(d, False)]
        av.available[name] = not miss and not deps
        if miss:
            av.missing_bands[name] = miss
        if deps:
            av.blocked_by[name] = deps
    return av
