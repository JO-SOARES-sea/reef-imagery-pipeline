"""O recorte Sentinel-2 fica na grelha nativa (píxeis inteiros), sem rede."""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.windows import transform as window_transform

from scripts.download_s2_zone import native_window

TILE = from_origin(499980.0, 4100040.0, 10.0, 10.0)  # canto de um tile Sentinel-2 (múltiplos de 10 m)


def test_window_is_whole_native_pixels_and_covers_the_bounds():
    bounds = (549748.4, 4098117.0, 564041.7, 4108427.04)  # cantos fora da grelha, como os da Pedra do Valado
    w = native_window(bounds, TILE)
    assert all(float(v).is_integer() for v in (w.col_off, w.row_off, w.width, w.height))
    t = window_transform(w, TILE)
    assert t.c % 10 == 0 and t.f % 10 == 0
    assert t.c <= bounds[0] and t.f >= bounds[3]
    assert t.c + w.width * 10 >= bounds[2] and t.f - w.height * 10 <= bounds[1]


def test_reading_the_window_returns_the_native_pixels_at_their_true_position():
    data = (np.arange(60)[None, :] + 1000 * np.arange(60)[:, None]).astype("float32")
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", width=60, height=60, count=1, dtype="float32", crs="EPSG:32629",
                      transform=TILE) as ds:
            ds.write(data, 1)
        with mem.open() as src:
            w = native_window((499980 + 84.0, 4100040 - 300.0, 499980 + 384.0, 4100040 - 70.4), src.transform)
            arr = src.read(1, window=w)
            t = window_transform(w, src.transform)
    # the first pixel's value names its native row/column, and its label must point at that same pixel
    row, col = divmod(int(arr[0, 0]), 1000)
    assert (t.c, t.f) == rasterio.transform.xy(TILE, row, col, offset="ul")
