# Reef Imagery Pipeline
### Satellite-Derived Bathymetry & Underwater Visibility Prediction — Algarve Coast

[![Live Presentation](https://img.shields.io/badge/Presentation-Live-brightgreen?logo=github)](https://JO-SOARES-sea.github.io/reef-imagery-pipeline/)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?logo=open-source-initiative)](https://opensource.org/licenses/MIT)
[![Data: Sentinel-2 L2A](https://img.shields.io/badge/Data-Sentinel--2%20L2A-orange)](https://dataspace.copernicus.eu/)
[![Calibration: ICESat-2](https://img.shields.io/badge/Calibration-ICESat--2-purple)](https://icesat-2.gsfc.nasa.gov)
[![Institution: NOVA IMS](https://img.shields.io/badge/Institution-NOVA%20IMS-black)](https://www.novaims.unl.pt)
[![Tests](https://img.shields.io/badge/Tests-374%20passing-brightgreen)]()

**Sea Project · NOVA IMS — Information Management School**
*Soares · 2025–2026*

---

## Overview

This pipeline transforms freely available **Sentinel-2 L2A** imagery into actionable marine-science outputs for eight reef complexes along the Portuguese Algarve coast (Faro → Carvoeiro, 0–20 m depth domain).

| Output | Description |
|---|---|
| **Satellite-Derived Bathymetry (SDB)** | Stumpf log-ratio depth maps calibrated to IH/DGRM isobaths |
| **Bottom Visibility Index (BVI)** | Random Forest score (0–1) combining Beer-Lambert optics, SNR, terrain exposure |
| **Scene Usability Filter** | IPMA sea-state + Sentinel-1 roughness gating to reject storm-degraded acquisitions |
| **Drift Monitoring** | Feature-drift alerts and HTML trend reports for continuous model health tracking |

The physics core uses Gordon/QAA Kd490 inversion, Snell refraction, and Beer-Lambert two-way transmittance. An optional ICESat-2 validation module provides independent depth ground-truth.

---

## Quick Start

```bash
# 1. Create and activate environment
python3.11 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies (all extras live in pyproject.toml)
pip install -e ".[dev]"          # core deps + pytest, ruff, mypy, black

# 3. Run the pipeline (simulated mode, target depth 16 m)
python -m src.orchestrator_run --depth 16.0

# 4. Launch the Flask dashboard
python dashboard/app.py          # → http://localhost:5000

# 5. Offline test suite (no satellite credentials needed)
pytest tests/ -m "not network" -q
```

---

## Repository Structure

```
reef-imagery-pipeline/
│
├── src/                              # Core pipeline modules
│   ├── constants.py                  # All physical constants (single source of truth)
│   ├── orchestrator_run.py           # Top-level pipeline entry point
│   ├── reef_ml_predictor_acolite.py  # Kd inversion · Stumpf SDB · BVI features
│   ├── ranking_model.py              # Siamese ranker + RF predict_score()
│   ├── bathy_calibrator.py           # IH/DGRM isobath regression
│   ├── coastal_topography.py         # GLO-30 / DGT LiDAR terrain features
│   ├── drift_monitor.py              # Feature drift detection (shadow layer)
│   ├── ipma_sea_state.py             # IPMA wave/wind scene filter
│   ├── sentinel1_roughness.py        # Sentinel-1 GRD sigma0 sea-roughness
│   ├── cmems_kd490.py                # Live CMEMS Kd490 climatology
│   ├── icesat2_validation.py         # ICESat-2 ATL03 depth validation
│   ├── stac_ingest.py                # STAC scene discovery
│   ├── sensor_config.py              # Multi-sensor band registry
│   └── utils.py                      # Raster I/O · Snell refraction · Beer-Lambert
│
├── dashboard/                        # Flask + Leaflet.js dashboard
│   ├── app.py                        # Flask routes & tile enhancement
│   ├── index.html                    # Leaflet map frontend
│   └── dashboard_layers.json         # Layer catalogue served to the UI
│
├── scripts/                          # Standalone analysis & utility scripts
│   ├── reef_imagery_pipeline_v3.py   # Orchestrator v3 (multi-step)
│   ├── predict_bathy_ml.py           # BVI prediction at a GPS point
│   ├── zimbral_best_visibility.py    # Best-visibility scene finder
│   ├── generate_reef_map.py          # Reef map generation utility
│   └── ...
│
├── models/                           # Trained ML artefacts
│   ├── bvi_model.pkl                 # Random Forest BVI regressor
│   ├── bvi_weights.json              # Feature weights
│   └── visibility_rf_bathy.pkl       # Visibility RF (legacy)
│
├── data/                             # Reference & training data
│   ├── raw/                          # Unprocessed inputs
│   ├── processed/                    # Feature matrices & labels
│   └── local/                        # Auto-generated local CSVs (gitignored)
│
├── docs/                             # Extended documentation
│   ├── BATHYMETRY_DOCUMENTATION.md
│   ├── DGT_SENTINEL_INTEGRATION.md
│   ├── DGT_STAC_GUIDE.md
│   ├── IMPLEMENTATION_CHECKLIST.md
│   ├── README_v3.md
│   ├── README_bathy.md
│   └── README_DGT_INTEGRATION.md
│
├── tests/                            # pytest test suite (374 tests, 0 warnings)
├── outputs/                          # Generated GeoTIFFs & reports (gitignored)
├── pyproject.toml                    # Build config + optional extras (cmems, dev)
├── CONTRIBUTING.md
└── CLAUDE.md                         # AI collaboration guidelines
```

---

## Pipeline Data Flow

```
Sentinel-2 L2A
    │
    ▼ ACOLITE BOA atmospheric correction
    ▼ Gordon/QAA Kd490 spectral inversion
    ▼ Stumpf (2003) log-ratio SDB
    ▼ IH/DGRM isobath calibration  ──→  depth GeoTIFF
    ▼ Beer-Lambert two-way transmittance
    ▼ Random Forest BVI score
    ▼ Drift monitoring (shadow layer)
    ▼
    Flask dashboard  ──→  Leaflet.js map
```

---

## Study Area

Eight reef complexes along the Algarve coast, Portugal (36.9°N – 37.2°N, 0–20 m depth):

| Site | Latitude | Longitude |
|---|---|---|
| Pedra de Santa Eulália | 37.0691 | −8.2102 |
| Zimbral | 36.9636 | −7.9356 |
| Baixa dos Cimbres | 37.1200 | −8.5800 |
| + 5 additional sites | … | … |

---

## Key Technologies

| Layer | Technology |
|---|---|
| Imagery | Sentinel-2 L2A (10 m, ESA / Copernicus Data Space) |
| Atmospheric correction | ACOLITE / Sen2Cor simulation |
| Bathymetry algorithm | Stumpf (2003) log-ratio SDB |
| Calibration data | IH/DGRM isobaths · ICESat-2 ATL03 |
| DEM | DGT MDT-50cm LiDAR · Copernicus GLO-30 |
| ML scoring | scikit-learn Random Forest |
| Scene filtering | IPMA ocean API · Sentinel-1 GRD |
| Dashboard | Flask + Leaflet.js |
| Tests | pytest · 374 offline tests · 0 warnings |

---

## Testing

```bash
# Offline suite — no credentials required
pytest tests/ -m "not network" -q

# With coverage report
pytest tests/ -m "not network" --cov=src -q

# Network-dependent tests (requires CDSE / Planetary Computer credentials)
pytest tests/ -m "network"
```

---

## Configuration

Optional environment variables for live data sources:

| Variable | Purpose |
|---|---|
| `CMEMS_USER` / `CMEMS_PASSWORD` | Live Kd490 climatology from CMEMS |
| `DGT_CDD_USERNAME` / `DGT_CDD_PASSWORD` | DGT 50cm LiDAR tiles (Copernicus Data Space) |
| `PC_SDK_SUBSCRIPTION_KEY` | Enhanced Planetary Computer STAC access |

---

## Documentation

| Document | Description |
|---|---|
| [docs/README_v3.md](docs/README_v3.md) | Pipeline v3 architecture and changelog |
| [docs/README_bathy.md](docs/README_bathy.md) | Bathymetry calibration methodology |
| [docs/DGT_STAC_GUIDE.md](docs/DGT_STAC_GUIDE.md) | DGT STAC API guide (Portuguese) |
| [docs/DGT_SENTINEL_INTEGRATION.md](docs/DGT_SENTINEL_INTEGRATION.md) | Sentinel + DGT data integration |
| [docs/BATHYMETRY_DOCUMENTATION.md](docs/BATHYMETRY_DOCUMENTATION.md) | Full bathymetry science reference |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines |

---

## Citation

```bibtex
@misc{soares2026reef,
  author    = {João Soares},
  title     = {Reef Imagery Pipeline: Satellite-Derived Bathymetry and
               Underwater Visibility Prediction for the Algarve Coast},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/JO-Soares-Sea/reef-imagery-pipeline},
  note      = {Master's Project, NOVA IMS -- Information Management School}
}
```

---

*NOVA IMS — Information Management School, Universidade Nova de Lisboa*
