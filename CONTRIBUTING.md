# Contributing to Reef Imagery Pipeline

## Code Style
- Use Python 3.11+ (required by the `scikit-learn==1.8.0` pin)
- Follow PEP 8 conventions
- Use type hints where possible
- Maximum line length: 120 characters

## Development Workflow

1. **Clone and setup**
   ```bash
   git clone <repo>
   cd reef_imagery_pipeline
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   # Optional development extras:
   # pip install -e .[dev]
   ```

2. **Make changes on a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Run tests and linting**
   ```bash
   pytest tests/ -q                          # full suite (207+ tests)
   pytest tests/ --cov=src --cov-report=term # with coverage
   flake8 src/ tests/ --max-line-length=127
   black --check src/ tests/ --line-length=127
   mypy src/coastal_topography.py src/ranking_model.py src/drift_monitor.py --ignore-missing-imports
   ```

4. **Commit with clear messages**
   - Use conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`
   - Keep commits focused and atomic

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Project Structure

```
reef_imagery_pipeline/
├── src/
│   ├── coastal_topography.py    # CoastalTopographyAnalyzer — GLO-30/DGT DEM features
│   ├── dgt_sentinel_integrator.py  # DGT MDT-50cm + Sentinel-2 stack
│   ├── ranking_model.py         # predict_score() + terrain_exposure_modifier()
│   ├── drift_monitor.py         # Feature drift, estimate_plume_extent()
│   ├── orchestrator_run.py      # Full pipeline orchestrator
│   └── ...                      # Other physics/ML modules
├── tests/
│   ├── test_coastal_topography.py   # Phase 3-5 terrain tests
│   ├── test_ranking_model.py    # BVI scoring + terrain modifier tests
│   ├── test_drift_monitor.py    # Drift + plume estimation tests
│   └── ...
├── outputs/coastal_topography/  # Pre-computed 15-site terrain features
├── dashboard/                   # Flask + Leaflet dashboard
├── .github/workflows/ci.yml     # CI: full test suite + lint + mypy + security
├── requirements.txt             # Dependencies (includes geopandas, rioxarray)
└── README.md
```

### DGT / GLO-30 DEM Integration

The `CoastalTopographyAnalyzer` supports three DEM sources:

| `dem_source` | Data | Auth required |
|:--|:--|:--|
| `"dgt"` | DGT MDT-50cm (0.5 m LiDAR) | Yes — DGT S3 credentials |
| `"copernicus"` | Copernicus GLO-30 via CDSE (30 m) | Yes — `~/.copernicusmarine` |
| `"srtm"` | Copernicus GLO-30 public AWS (30 m) | No |
| `"auto"` | DGT → CDSE → public (best available) | Optional |

For DGT access, contact [DGT/SNIG](https://snig.dgterritorio.gov.pt/) to request
MDT-50cm S3 credentials.

## Key Files Edited

- `orchestrator.py` - Updated paths for reef_Output_Master consolidation
- `orchestrator_run.py` - Updated paths for reef_Output_Master consolidation
- `run_benthic_physics_comparison.py` - Updated paths for reef_Output_Master consolidation
- `.gitignore` - Enhanced with virtualenv, IDE, cache patterns

## Testing

The CI/CD pipeline runs:
- Python syntax checks (py_compile)
- Linting (flake8)
- Code formatting (black)
- Type checking (mypy)
- Security scanning (bandit, safety)
- Dependency verification

## Questions?

See README.md, README_v2.md, README_v3.md for documentation on specific pipelines.
