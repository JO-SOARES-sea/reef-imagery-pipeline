import pandas as pd
import pytest

from src.ranking_model import predict_score


@pytest.mark.network
def test_fft_cleanliness():
    import planetary_computer as pc
    from pystac_client import Client

    from src.reef_ml_predictor import extract_features_from_stac

    catalog = Client.open(
        'https://planetarycomputer.microsoft.com/api/stac/v1',
        modifier=pc.sign_inplace,
    )
    search = catalog.search(
        collections=['sentinel-2-l2a'],
        intersects={'type': 'Point', 'coordinates': [-8.20978, 37.05811]},
        datetime="2020-01-01/2026-05-20",
        query={"eo:cloud_cover": {"lt": 5}}
    )
    items = list(search.items())
    data = []
    for item in items:
        props = item.properties
        if props.get('s2:nodata_pixel_percentage', 100) > 20:
            continue
        data.append({
            'date_str': item.datetime.strftime('%Y-%m-%d'),
            'datetime': item.datetime,
            'cloud_cover': props.get('eo:cloud_cover', 100),
            'sun_elevation': props.get('view:sun_elevation', 45)
        })
    df = pd.DataFrame(data)
    df['physics_score'] = df.apply(
        lambda r: predict_score(extract_features_from_stac(r, 16.0))["score"], axis=1
    )
    df = df.sort_values('physics_score', ascending=False).drop_duplicates('date_str')
    result = df[df['date_str'] == '2025-09-25']
    assert not result.empty, "Expected at least one record for 2025-09-25"
    assert "physics_score" in result.columns
    assert result["physics_score"].iloc[0] >= 0
