"""
src/navigation/lowrance_sl3.py — Leitor mínimo de registos de sonda Lowrance SL3 (sem dependências externas).

Lê, frame a frame, os campos de navegação e de sonda de um ficheiro `.sl3` (Elite FS / HDS):
posição GPS, hora, profundidade da água, temperatura e canal. Serve para extrair batimetria própria do barco ao longo
dos trilhos (verdade de relevo a 13–20 m, onde os levantamentos públicos não chegam).

Formato (verificado nos ficheiros de 2024-09 do Rui):
  - cabeçalho do ficheiro: 8 bytes, uint16 formato (3 = SL3), uint16 versão, uint16 blocksize, uint16 reservado;
  - frames encadeados; cada frame começa com uint32 = a sua própria posição no ficheiro e tem o tamanho em uint16 @ +8;
  - campos usados (little-endian): canal uint32 @ +12; hora Unix uint32 @ +40; profundidade float32 @ +48 (pés);
    temperatura float32 @ +88 (°C); x, y int32 @ +92/+96 em "Lowrance-Mercator" (raio polar WGS84 6 356 752,3142 m).
"""

from __future__ import annotations

import math
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

SL3_FORMAT = 3
_R_POLAR = 6356752.3142
_FT_TO_M = 0.3048


@dataclass(frozen=True)
class SL3Frame:
    offset: int
    channel: int
    time_unix: int
    depth_m: float
    temp_c: float
    lat: float
    lon: float


def mercator_to_latlon(x: int, y: int) -> tuple[float, float]:
    """Converte as coordenadas inteiras Lowrance-Mercator em latitude/longitude (graus)."""
    lon = math.degrees(x / _R_POLAR)
    lat = math.degrees(2 * math.atan(math.exp(y / _R_POLAR)) - math.pi / 2)
    return lat, lon


def iter_frames(path: str | Path, channels: set[int] | None = None) -> Iterator[SL3Frame]:
    """Itera os frames de um ficheiro SL3; ``channels`` filtra por canal (None = todos)."""
    with open(path, "rb") as fh:
        head = fh.read(8)
        fmt = struct.unpack("<H", head[:2])[0]
        if fmt != SL3_FORMAT:
            raise ValueError(f"{path}: formato {fmt}, esperado SL3 ({SL3_FORMAT})")
        pos = 8
        while True:
            fh.seek(pos)
            hdr = fh.read(100)
            if len(hdr) < 100:
                return
            off, = struct.unpack_from("<I", hdr, 0)
            size, = struct.unpack_from("<H", hdr, 8)
            if off != pos or size < 100:
                return  # fim do registo ou frame corrompido
            channel, = struct.unpack_from("<I", hdr, 12)
            if channels is None or channel in channels:
                t, = struct.unpack_from("<I", hdr, 40)
                depth_ft, = struct.unpack_from("<f", hdr, 48)
                temp, = struct.unpack_from("<f", hdr, 88)
                x, y = struct.unpack_from("<ii", hdr, 92)
                lat, lon = mercator_to_latlon(x, y)
                yield SL3Frame(pos, channel, t, depth_ft * _FT_TO_M, temp, lat, lon)
            pos += size
