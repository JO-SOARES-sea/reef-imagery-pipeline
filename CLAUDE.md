# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"   # includes pytest, ruff, mypy, black

# Run tests (offline only — skip network-dependent tests)
pytest tests/ -m "not network" -q
pytest tests/test_ranking_model.py -q              # single test file
pytest tests/ -m "not network" --cov=src -q        # with coverage

# Lint / format
flake8 src/ tests/ --select=E9,F63,F7,F82          # blocking syntax errors only
black src/ tests/ --line-length 127
ruff check src/

# Type check (non-blocking in CI)
mypy src/coastal_topography.py src/ranking_model.py src/drift_monitor.py --ignore-missing-imports

# Run the pipeline
python -m src.orchestrator_run --depth 16.0
python -m src.orchestrator_run --depth 16.0 --config config.yaml

# BVI prediction at a point
python scripts/predict_bathy_ml.py --lon -8.2103 --lat 37.069 --json

# Flask dashboard
python dashboard/app.py
```

## Architecture

The pipeline converts Sentinel-2 L2A imagery into satellite-derived bathymetry (SDB) and a Bottom Visibility Index (BVI) for 8 reef complexes along the Algarve coast (Faro → Carvoeiro, 0–20 m depth domain).

### Data flow

```
Sentinel-2 L2A  →  ACOLITE BOA correction  →  Gordon/QAA Kd inversion
    →  Stumpf log-ratio SDB  →  IH/DGRM isobath calibration  →  depth GeoTIFF
    →  Beer-Lambert transmittance  →  Random Forest BVI score
    →  Drift monitoring (shadow layer)  →  Flask dashboard
```

### Core modules (`src/`)

| Module | Role |
|---|---|
| `constants.py` | All physical constants (Stumpf coefficients, Kd490 table, Beer-Lambert params) — single source of truth |
| `reef_ml_predictor_acolite.py` | Gordon/QAA Kd inversion, Stumpf SDB depth map, `run_predictor()` entry point |
| `orchestrator_run.py` | Top-level pipeline orchestrator; delegates all physics to `run_predictor()` |
| `bathy_calibrator.py` | IH/DGRM isobath ingestion, Stumpf m0/m1 regression, depth zone classification |
| `stumpf_emodnet_calibration.py` | EMODnet DTM reprojection, Stumpf-EMODnet regression fallback |
| `ranking_model.py` | Siamese ranker + RF `predict_score()`, `terrain_exposure_modifier()` |
| `coastal_topography.py` | `CoastalTopographyAnalyzer`: GLO-30/DGT DEM → slope/aspect/exposure features |
| `drift_monitor.py` | Feature drift detection, `estimate_plume_extent()`, z-score alerts — never blocks inference |
| `stac_ingest.py` | STAC scene discovery (Earth Search, Planetary Computer, DGT STAC) |
| `ih_bathy_features.py` | `BathyFeatureEngine` — bathymetry-derived ML features |
| `enhancer.py` | SNR-adaptive NLM denoising, CLAHE sharpening |
| `utils.py` | Raster I/O, Snell refraction, Beer-Lambert transmittance, `get_kd490()`, `build_coastal_geojson()` |
| `sensor_config.py` | Per-sensor band configs (Sentinel-2, PlanetScope SuperDove) |

### ML models (`models/`)

Trained artifacts: `bvi_model.pkl` (RF regressor), `bvi_weights.json`, `visibility_rf_bathy.pkl`. Loaded lazily inside `predict_score()` — not re-imported on each call.

### Terrain BVI modifier

`terrain_exposure_modifier(slope_mean, aspect_mean)` in `ranking_model.py` returns a multiplier in `[0.5, 1.0]` based on the site's orientation relative to the dominant SW swell (225°). Pre-computed features for 15 Algarve sites are at `outputs/coastal_topography/algarve_coastal_features.csv`.

### Coastal features artefact flow

The dashboard endpoint `/api/coastal-features` reads **`outputs/coastal_topography/algarve_coastal_features.geojson`** — it never reads the CSV directly. Always keep the GeoJSON in sync with the CSV.

Two paths that regenerate the pair:

| Path | When to use | GeoJSON auto-generated? |
|---|---|---|
| `python scripts/generate_coastal_features_dgt.py` | Local run with DGT CDD creds | **Yes** — calls `build_coastal_geojson()` at the end of `main()` |
| VM extraction via JupyterHub | LiDAR tiles IP-restricted to VM | **No** — run `python scripts/build_coastal_geojson.py` manually after updating the CSV |

`build_coastal_geojson(csv_path, geojson_path)` lives in `src/utils.py`. It writes atomically (temp→rename) and creates a `.bak_YYYYMMDDTHHMMSS` backup before overwriting. The CLI wrapper at `scripts/build_coastal_geojson.py` takes `--csv` and `--out` flags.

```bash
# After a VM extraction — parse CSV:: lines from stdout → update CSV — then:
python scripts/build_coastal_geojson.py
```

### DEM fallback chain

DGT MDT-50cm (50 cm LiDAR, requires S3 credentials) → Copernicus GLO-30 (30 m, public AWS) → SRTM. The `CoastalTopographyAnalyzer` `dem_source` parameter controls this: `"dgt"`, `"copernicus"`, `"srtm"`, or `"auto"`.

### Dashboard

`dashboard/app.py` is a Flask server serving `dashboard/index.html` (Leaflet map). It reads GeoTIFF outputs and cached BVI JSON via local routes; `enhance_local_tile()` in app.py applies CLAHE/NLM inline rather than importing `src/enhancer.py`.

### Test markers

Tests that require live network/satellite access are marked `@pytest.mark.network`. Run offline suite with `-m "not network"`.

## Key conventions

- All physical constants live in `src/constants.py`. Never hardcode Stumpf coefficients, Kd490 values, or SNR thresholds elsewhere.
- `drift_monitor` is a **shadow layer** — it must never raise exceptions that block `predict_score()`. Wrap all calls in try/except.
- Optional heavy dependencies (geopandas, rioxarray, rasterstats) are guarded with `try/import` blocks throughout `src/`. Keep this pattern when adding new optional deps.
- Black line length is **127** (not the default 88) — matches the CI flake8 `--max-line-length=127`.
- The `@pytest.mark.network` marker must be applied to any test that hits external STAC/satellite APIs.
