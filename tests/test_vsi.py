import planetary_computer as pc
import pytest
import rasterio
from pyproj import Transformer
from pystac_client import Client
from rasterio.windows import from_bounds

lat, lon = 37.05811, -8.20978


@pytest.mark.network
def test_vsi_b02_read():
    catalog = Client.open('https://planetarycomputer.microsoft.com/api/stac/v1', modifier=pc.sign_inplace)
    search = catalog.search(
        collections=['sentinel-2-l2a'],
        intersects={'type': 'Point', 'coordinates': [lon, lat]},
        datetime="2024-09-30/2024-09-30"
    )
    item = list(search.items())[0]
    b02_href = item.assets["B02"].href
    with rasterio.Env(AWS_NO_SIGN_REQUEST='YES', GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'), rasterio.open(b02_href) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        x, y = transformer.transform(lon, lat)
        window = from_bounds(x - 500, y - 500, x + 500, y + 500, src.transform)
        data = src.read(1, window=window)
        assert data.shape[0] > 0 and data.shape[1] > 0
        assert data.mean() >= 0
