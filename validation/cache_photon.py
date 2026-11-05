"""
cache_photon.py — Local cache for ICESat-2 ATL03/ATL08 photon downloads.

OpenAltimetry requests are slow (5–30 s per track) and the Algarve tracks
repeat on a 91-day cycle.  Caching avoids re-downloading photons for the
same bbox + date window on every validation run.

Cache key: SHA-256 of (minx, miny, maxx, maxy, date_start, date_end, track_id).
Storage: JSON files under data/cache_icesat2/.

Usage:
    from validation.cache_photon import PhotonCache

    cache = PhotonCache()
    photons = cache.get(bbox, date_start, date_end, track_id)
    if photons is None:
        photons = fetch_from_api(...)
        cache.put(bbox, date_start, date_end, track_id, photons)
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path("data/cache_icesat2")


class PhotonCache:
    """
    Persistent JSON cache for ATL03/ATL08 photon records from OpenAltimetry.

    Args:
        cache_dir: Directory where cache files are stored.
    """

    def __init__(self, cache_dir: str | Path = _DEFAULT_CACHE_DIR) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Key derivation
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(
        bbox: tuple[float, float, float, float],
        date_start: str,
        date_end: str,
        track_id: int | None,
    ) -> str:
        raw = f"{bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f}|{date_start}|{date_end}|{track_id}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        bbox: tuple[float, float, float, float],
        date_start: str,
        date_end: str,
        track_id: int | None = None,
    ) -> list[dict[str, Any]] | None:
        """
        Return cached photon records or None on cache miss.

        Args:
            bbox:       (minx, miny, maxx, maxy) in WGS84 decimal degrees.
            date_start: ISO date string "YYYY-MM-DD".
            date_end:   ISO date string "YYYY-MM-DD".
            track_id:   ICESat-2 reference ground track ID (1–1387), or None for all.

        Returns:
            List of photon dicts or None.
        """
        key  = self._make_key(bbox, date_start, date_end, track_id)
        path = self._path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            log.debug("PhotonCache hit: %s (%d records)", key, len(data))
            return data
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("PhotonCache corrupt entry %s: %s — deleting", key, exc)
            path.unlink(missing_ok=True)
            return None

    def put(
        self,
        bbox: tuple[float, float, float, float],
        date_start: str,
        date_end: str,
        track_id: int | None,
        photons: list[dict[str, Any]],
    ) -> None:
        """Persist photon records to cache."""
        key  = self._make_key(bbox, date_start, date_end, track_id)
        path = self._path(key)
        path.write_text(json.dumps(photons, indent=None))
        log.debug("PhotonCache stored: %s (%d records)", key, len(photons))

    def invalidate(
        self,
        bbox: tuple[float, float, float, float],
        date_start: str,
        date_end: str,
        track_id: int | None = None,
    ) -> bool:
        """Remove a cache entry. Returns True if the entry existed."""
        key  = self._make_key(bbox, date_start, date_end, track_id)
        path = self._path(key)
        if path.exists():
            path.unlink()
            log.info("PhotonCache invalidated: %s", key)
            return True
        return False

    def clear(self) -> int:
        """Delete all cache entries. Returns count of deleted files."""
        count = 0
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
            count += 1
        log.info("PhotonCache cleared: %d entries deleted", count)
        return count

    def stats(self) -> dict[str, Any]:
        files = list(self.cache_dir.glob("*.json"))
        total_mb = sum(f.stat().st_size for f in files) / 1e6
        return {"entries": len(files), "size_mb": round(total_mb, 2)}
