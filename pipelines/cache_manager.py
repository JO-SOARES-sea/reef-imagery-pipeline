"""
cache_manager.py — Filesystem cache for Sentinel-2 and BOA reflectance tiles.

Prevents re-downloading or re-processing imagery that is already on disk.
Cache keys are derived from (site, date, bbox) so each unique acquisition
gets its own slot.

Usage:
    from pipelines.cache_manager import S2Cache

    cache = S2Cache()
    if cache.has_boa(site="pedra_do_alto", date="2024-09-30"):
        bands = cache.load_boa(site="pedra_do_alto", date="2024-09-30")
    else:
        bands = ... # download + process
        cache.save_boa(site="pedra_do_alto", date="2024-09-30", bands=bands)
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

_DEFAULT_ROOT = Path("data")


class S2Cache:
    """
    Two-tier cache:
      data/cache_s2/   — raw Sentinel-2 scene TIFs (large, ~500 MB each)
      data/cache_boa/  — atmospherically corrected BOA reflectance arrays (small .npy)

    Args:
        root: Base directory for all cache subdirectories.
    """

    def __init__(self, root: str | Path = _DEFAULT_ROOT) -> None:
        self.root   = Path(root)
        self.s2_dir = self.root / "cache_s2"
        self.boa_dir= self.root / "cache_boa"
        self.s2_dir.mkdir(parents=True, exist_ok=True)
        self.boa_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(site: str, date: str) -> str:
        return f"{site}__{date}"

    # ------------------------------------------------------------------
    # Raw S2 scene cache
    # ------------------------------------------------------------------

    def s2_dir_for(self, site: str, date: str) -> Path:
        """Return the cache directory for a raw S2 scene (may not exist yet)."""
        return self.s2_dir / self._key(site, date)

    def has_s2(self, site: str, date: str) -> bool:
        d = self.s2_dir_for(site, date)
        return d.exists() and any(d.glob("*.tif"))

    def s2_tifs(self, site: str, date: str) -> list[Path]:
        return sorted(self.s2_dir_for(site, date).glob("*.tif"))

    def invalidate_s2(self, site: str, date: str) -> None:
        d = self.s2_dir_for(site, date)
        if d.exists():
            shutil.rmtree(d)
            log.info("S2 cache invalidated: %s", d)

    # ------------------------------------------------------------------
    # BOA reflectance cache  (.npy per band + metadata .json)
    # ------------------------------------------------------------------

    def _boa_dir(self, site: str, date: str) -> Path:
        return self.boa_dir / self._key(site, date)

    def has_boa(self, site: str, date: str) -> bool:
        d = self._boa_dir(site, date)
        return d.exists() and (d / "meta.json").exists()

    def save_boa(
        self,
        site: str,
        date: str,
        bands: dict[str, np.ndarray],
        meta: dict[str, Any] | None = None,
    ) -> None:
        """
        Persist BOA reflectance arrays.

        Args:
            site:   Site identifier (e.g. "pedra_do_alto").
            date:   Scene date string (e.g. "2024-09-30").
            bands:  Dict mapping band name → numpy array (e.g. {"B02": arr, ...}).
            meta:   Optional metadata dict (CRS, transform, etc.) persisted as JSON.
        """
        d = self._boa_dir(site, date)
        d.mkdir(parents=True, exist_ok=True)
        for band_name, arr in bands.items():
            np.save(d / f"{band_name}.npy", arr)
        record = {"site": site, "date": date, "bands": list(bands.keys()), **(meta or {})}
        (d / "meta.json").write_text(json.dumps(record, indent=2))
        log.info("BOA cache saved: %s  bands=%s", d, list(bands.keys()))

    def load_boa(self, site: str, date: str) -> tuple[dict[str, np.ndarray], dict]:
        """
        Load cached BOA arrays.

        Returns:
            (bands_dict, meta_dict)
        """
        d = self._boa_dir(site, date)
        if not self.has_boa(site, date):
            raise FileNotFoundError(f"BOA cache miss: {d}")
        meta = json.loads((d / "meta.json").read_text())
        bands = {b: np.load(d / f"{b}.npy") for b in meta["bands"]}
        log.info("BOA cache hit: %s  bands=%s", d, list(bands.keys()))
        return bands, meta

    def invalidate_boa(self, site: str, date: str) -> None:
        d = self._boa_dir(site, date)
        if d.exists():
            shutil.rmtree(d)
            log.info("BOA cache invalidated: %s", d)

    # ------------------------------------------------------------------
    # Cache stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        s2_count  = sum(1 for _ in self.s2_dir.iterdir() if _.is_dir())
        boa_count = sum(1 for _ in self.boa_dir.iterdir() if _.is_dir())

        def _dir_mb(p: Path) -> float:
            return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6

        return {
            "s2_scenes":  s2_count,
            "boa_scenes": boa_count,
            "s2_mb":      round(_dir_mb(self.s2_dir),  1),
            "boa_mb":     round(_dir_mb(self.boa_dir), 1),
        }
