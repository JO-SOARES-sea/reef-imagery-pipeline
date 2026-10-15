"""
src/s2_radiometry.py — Explicit, evidence-based DN → reflectance conversion for Sentinel-2 L2A.

Policy (no blind corrections)
-----------------------------
* The conversion for an input is decided ONLY from explicit metadata:
    1. the STAC asset's ``raster:bands`` ``scale`` + ``offset`` (Earth Search v1 publishes these), or
    2. the L2A product metadata (``MTD_MSIL2A.xml``) ``QUANTIFICATION_VALUE`` + ``BOA_ADD_OFFSET``
       for the band (ESA, processing baseline >= 04.00), or
    3. a processing baseline < 04.00 together with an explicit quantification value (no offset existed).
* Anything else is ``radiometry_unknown`` and must be blocked from scientific use. We never assume
  that two providers deliver equivalent values just because they share an acquisition, and we never
  apply a -1000 offset "just in case".
* The conversion is applied exactly once. Inputs that are already floating point are refused,
  because a float raster is either already reflectance or of unknown provenance — converting it
  again would double-correct.
* ``nodata`` (DN 0 for Sentinel-2 L2A) is masked BEFORE any arithmetic. Valid negative reflectances
  (DN in [1, |offset|) for baseline >= 04.00) are preserved, not clipped.

Evidence used when this module was written (2026-10-09):
* Earth Search README: "If the `offset` is anything other than 0, then it should be applied after the
  scale factor to correct the data."  Item S2B_29SNB_20250925_0_L2A publishes scale=0.0001,
  offset=-0.1, nodata=0 for blue/green/red/nir.
* ESA L2A product page: "the dynamic range of the Level-2A products is shifted by a band-dependent
  constant: BOA_ADD_OFFSET ... initially set to -1000 Digital counts ... DN=0 remains the NO_DATA value".
* Planetary Computer item for the same datatake exposes no raster:bands scale/offset on B02/B04/B08;
  Microsoft's baseline-change notebook states the offset "propagates to the L2A data hosted by the
  Planetary Computer". → without product metadata, a PC asset is radiometry_unknown here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

RADIOMETRY_KNOWN = "known"
RADIOMETRY_UNKNOWN = "radiometry_unknown"

# Sentinel-2 L2A: DN 0 is the NO_DATA value (ESA L2A product definition).
S2_L2A_NODATA_DN = 0


@dataclass(frozen=True)
class Radiometry:
    """How to turn one band's stored values into BOA reflectance — or why we cannot."""

    status: str  # RADIOMETRY_KNOWN | RADIOMETRY_UNKNOWN
    scale: float | None = None
    offset: float | None = None  # applied AFTER scale: refl = DN * scale + offset
    nodata: float | int | None = None
    source: str = ""  # where scale/offset came from
    evidence: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_known(self) -> bool:
        return self.status == RADIOMETRY_KNOWN

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "scale": self.scale,
            "offset": self.offset,
            "nodata": self.nodata,
            "source": self.source,
            "evidence": list(self.evidence),
        }


def unknown(reason: str, *evidence: str) -> Radiometry:
    return Radiometry(status=RADIOMETRY_UNKNOWN, source=reason, evidence=tuple(evidence))


def parse_processing_baseline(value: Any) -> float | None:
    """'05.11' → 5.11 ; 'N0511' → 5.11 ; None/garbage → None."""
    if value is None:
        return None
    s = str(value).strip()
    m = re.fullmatch(r"N?(\d{2})\.?(\d{2})", s)
    if not m:
        return None
    return float(f"{int(m.group(1))}.{m.group(2)}")


def radiometry_from_stac_asset(asset_fields: Mapping[str, Any], item_properties: Mapping[str, Any]) -> Radiometry:
    """
    Decide the conversion from a STAC asset dict (``asset.to_dict()`` or ``extra_fields``) and the
    item properties. Only ``raster:bands`` with BOTH scale and offset is accepted for baseline >= 04.00.
    """
    baseline = parse_processing_baseline(item_properties.get("s2:processing_baseline"))
    bands = asset_fields.get("raster:bands") or []
    rb = bands[0] if bands else {}
    nodata = rb.get("nodata", None)
    dtype = str(rb.get("data_type", "")).lower()

    if dtype.startswith("float"):
        return unknown(
            "asset is floating point — already scaled or of unknown provenance; refusing second conversion",
            f"raster:bands.data_type={dtype}",
        )

    scale = rb.get("scale")
    offset = rb.get("offset")
    ev = [f"s2:processing_baseline={item_properties.get('s2:processing_baseline')}",
          f"raster:bands={dict(rb)}"]
    if "earthsearch:boa_offset_applied" in item_properties:
        ev.append(f"earthsearch:boa_offset_applied={item_properties['earthsearch:boa_offset_applied']}")

    if scale is not None and offset is not None:
        return Radiometry(RADIOMETRY_KNOWN, float(scale), float(offset),
                          nodata if nodata is not None else S2_L2A_NODATA_DN,
                          "stac raster:bands scale+offset", tuple(ev))
    if scale is not None and baseline is not None and baseline < 4.0:
        # Pre-04.00 products had no BOA offset; an explicit scale is sufficient.
        return Radiometry(RADIOMETRY_KNOWN, float(scale), 0.0,
                          nodata if nodata is not None else S2_L2A_NODATA_DN,
                          "stac raster:bands scale, baseline < 04.00 (no offset existed)", tuple(ev))
    return unknown("no explicit scale+offset for this asset (baseline >= 04.00 or unknown)", *ev)


def radiometry_from_product_metadata(mtd_xml: str, band_id: int) -> Radiometry:
    """
    Parse ESA ``MTD_MSIL2A.xml`` for QUANTIFICATION_VALUE and the band's BOA_ADD_OFFSET.
    ``band_id`` is the ESA physical band index (B02→1, B03→2, B04→3, B08→7).
    """
    q = re.search(r"<BOA_QUANTIFICATION_VALUE[^>]*>\s*([0-9.]+)\s*<", mtd_xml)
    if not q:
        return unknown("MTD_MSIL2A.xml lacks BOA_QUANTIFICATION_VALUE")
    quant = float(q.group(1))
    off = re.search(rf'<BOA_ADD_OFFSET\s+band_id="{band_id}"\s*>\s*(-?[0-9.]+)\s*<', mtd_xml)
    baseline = re.search(r"<PROCESSING_BASELINE>\s*([0-9.]+)\s*<", mtd_xml)
    pb = parse_processing_baseline(baseline.group(1)) if baseline else None
    ev = (f"BOA_QUANTIFICATION_VALUE={quant}", f"PROCESSING_BASELINE={pb}")
    if off:
        add = float(off.group(1))
        return Radiometry(RADIOMETRY_KNOWN, 1.0 / quant, add / quant, S2_L2A_NODATA_DN,
                          "MTD_MSIL2A.xml BOA_ADD_OFFSET", ev + (f"BOA_ADD_OFFSET[{band_id}]={add}",))
    if pb is not None and pb < 4.0:
        return Radiometry(RADIOMETRY_KNOWN, 1.0 / quant, 0.0, S2_L2A_NODATA_DN,
                          "MTD_MSIL2A.xml, baseline < 04.00 (no offset)", ev)
    return unknown("MTD_MSIL2A.xml has no BOA_ADD_OFFSET for band and baseline is not < 04.00", *ev)


def dn_to_reflectance(dn: np.ndarray, rad: Radiometry, extra_nodata: float | int | None = None
                      ) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert stored integer DN to reflectance ONCE.

    Returns (reflectance float32 with NaN where invalid, valid bool mask).
    Raises ValueError for unknown radiometry or float input (would be a second conversion).
    """
    if not rad.is_known:
        raise ValueError(f"radiometry_unknown: {rad.source}")
    arr = np.asarray(dn)
    if np.issubdtype(arr.dtype, np.floating):
        raise ValueError("dn_to_reflectance expects stored integer DN; got float input (double-conversion risk)")
    valid = np.ones(arr.shape, dtype=bool)
    for nd in (rad.nodata, extra_nodata):
        if nd is not None:
            valid &= arr != nd
    refl = np.full(arr.shape, np.nan, dtype=np.float32)
    refl[valid] = (arr[valid].astype(np.float64) * rad.scale + rad.offset).astype(np.float32)
    return refl, valid
