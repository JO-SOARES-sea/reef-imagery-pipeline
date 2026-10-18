"""
Physical constants and calibration parameters for reef imagery pipeline.

References:
- Dierssen et al. (2003): Remote sensing of shallow coastal waters
- Stumpf et al. (2003): Determination of water depth with high-resolution
  satellite imagery over variable bottom types
- Gordon et al. (1988): Estimation of seawater constituents from radiance
"""

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

# Refractive index of water (Snell's law)
# Source: Standard value for seawater at 589nm (sodium D-line)
N_WATER = 1.333

# =============================================================================
# DEPTH AND OPTICAL LIMITS
# =============================================================================

# Maximum reliable depth for SDB (optical limit of blue/green ratio method)
# Beyond this, signal is dominated by water column, not bottom
SDB_OPTICAL_LIMIT_M = 40.0

# Default target depth for benthic visibility analysis (metres)
DEFAULT_DEPTH_TARGET = 16.0
DEPTH_TARGET = DEFAULT_DEPTH_TARGET  # Alias for backward compatibility

# =============================================================================
# STUMPF SDB ALGORITHM PARAMETERS
# =============================================================================

# Default Stumpf coefficients for Algarve oligotrophic waters (Kd≈0.045)
# Literature values from regional studies
# Stumpf formula: depth = m1 * ln(n*B02) / ln(n*B03) + m0
STUMPF_M0_DEFAULT = -16.0   # Intercept
STUMPF_M1_DEFAULT = 20.0    # Slope — used as the default in Stumpf depth calculation
STUMPF_N = 1000.0           # Log scaling factor

# Epsilon added inside the log arguments to prevent log(0).
# A single value across all Stumpf calibration modules avoids drift between
# implementations that each hardcoded 1e-6 independently.
STUMPF_LOG_EPSILON = 1e-6

# Literature m1 for Stumpf n=1000 in clear oligotrophic waters
# Range validated against Dierssen et al. 2003 and regional studies.
# STUMPF_M1_LITERATURE is a physically-grounded fixed slope used in
# single-isobath offset calibration (where we solve analytically for m0
# rather than regression). STUMPF_M1_DEFAULT is the interactive/default value.
STUMPF_M1_LITERATURE = 20.0

# Both constants are currently 20.0. This assertion fires if they diverge,
# which should only happen after deliberate recalibration — update the
# calibration logic accordingly if that occurs.
if STUMPF_M1_DEFAULT != STUMPF_M1_LITERATURE:
    raise ValueError(
        "STUMPF_M1_DEFAULT and STUMPF_M1_LITERATURE should be equal. "
        "If intentionally diverging (e.g. after local calibration), update "
        "the calibration logic that depends on each value accordingly."
    )

# =============================================================================
# ATTENUATION COEFFICIENTS (Kd490)
# =============================================================================

# Seasonal Kd490 table for Algarve coastal waters (m^-1)
# Based on historical in-situ measurements and satellite climatology
# Format: {month: Kd490_value}
KD490_TABLE = {
    1: 0.055,   # January — winter, moderate clarity
    2: 0.055,   # February — winter, moderate clarity
    3: 0.060,   # March — spring transition, moderate
    4: 0.065,   # April — spring, slightly turbid
    5: 0.070,   # May — spring bloom, more turbid
    6: 0.060,   # June — early summer, improving
    7: 0.045,   # July — summer, best clarity
    8: 0.042,   # August — summer, peak clarity
    9: 0.045,   # September — summer, excellent clarity
    10: 0.050,  # October — autumn, good clarity
    11: 0.055,  # November — autumn, moderate
    12: 0.055,  # December — winter, moderate
}
KD490_DEFAULT = 0.080

# =============================================================================
# GLINT CORRECTION PENALTIES (monthly factors)
# =============================================================================

# Monthly glint penalties for visibility score calculation
# Higher values = less penalty (better conditions)
# Lower values = more penalty (more sun glint risk)
GLINT_PENALTY = {
    1: 0.60,   # January — low sun, high glint risk
    2: 0.65,   # February
    3: 0.70,   # March — equinox, moderate glint
    4: 0.80,   # April
    5: 0.85,   # May — sun getting higher
    6: 0.90,   # June — solstice, sun highest
    7: 0.95,   # July — peak summer, lowest glint
    8: 0.95,   # August — peak summer
    9: 0.90,   # September — still good
    10: 0.75,  # October
    11: 0.65,  # November
    12: 0.60,  # December — solstice, high glint risk
}
# Default penalty for months not in table
GLINT_PENALTY_DEFAULT = 0.80

# =============================================================================
# QUALITY THRESHOLDS
# =============================================================================

# SNR threshold for acceptable quality
SNR_THRESHOLD = 3.0

# Cloud cover threshold (percent)
CLOUD_THRESHOLD = 5.0

# High uncertainty threshold for Kd (relative difference from prior)
KD_HIGH_UNCERTAINTY_THRESHOLD = 0.30  # 30%

# Per-pixel Kd490 map saturation ceiling (m^-1). When the Lee-2013 band-ratio
# map pins near its 2.0 clip (bright sand / sunglint), the field is physically
# meaningless and transmittance falls back to the scene-mean scalar kd_b02.
KD490_MAP_SATURATION_CEILING = 1.5

# =============================================================================
# CALIBRATION AND VALIDATION
# =============================================================================

# Isobath depths used for benthic zone classification (metres)
BENTHIC_ISOBATHS = [10, 20, 30]

# Context isobaths for zone classification only (metres)
CONTEXT_ISOBATHS = [50, 100]

# Pixel buffer around isobath vertices for sampling (pixels)
# ±3 pixels = ±30m at Sentinel-2 10m resolution
BUF_PIX = 3

# =============================================================================
# REFLECTANCE CONSTANTS (for benthic contrast calculation)
# =============================================================================

# Typical bottom reflectance values (BOA, 0-1 scale)
SAND_R = 0.25    # Bright sand
ROCK_R = 0.05    # Dark rock/reef

# Reflectance scaling factor for DN -> BOA conversion
# Sentinel-2 L2A products use 1/10000 scaling
REFLECTANCE_DN_SCALE = 10000.0

# DN threshold to detect unscaled vs already-scaled reflectance
REFLECTANCE_DN_THRESHOLD = 2.0

# Minimum FFT cleanliness score to avoid the heavy score penalty in ranking_model.
# Scores below this threshold indicate severe surface noise (glint / cloud shadow).
# The no-data proxy in reef_ml_predictor_acolite equals this value so that FFT-less
# runs are not penalised; callers that DO compute FFT should pass real values.
FFT_CLEAN_THRESHOLD = 5000

# =============================================================================
# VISIBILITY FORECAST (scripts/forecast_visibility.py)
# =============================================================================

# Calm-water horizontal visibility: Poole-Atkins Secchi depth (1.7 / Kd490),
# halved to approximate horizontal diver visibility.
VIS_POOLE_ATKINS = 1.7
VIS_HORIZONTAL_FACTOR = 0.5

# Reference depth for the near-bed orbital velocity (typical reef depth, m)
VIS_FORECAST_DEPTH_M = 12.0

# Near-bed orbital velocity at which fine sand / silt starts to resuspend (m/s)
VIS_RESUSPENSION_U_CRIT = 0.12

# Visibility divisor per unit of resuspension excess: vis = calm / (1 + k * excess)
VIS_RESUSPENSION_SENSITIVITY = 1.5

# Fraction of a day's resuspension still in the water column the next day
VIS_TURBIDITY_MEMORY = 0.5

# Rain runoff penalties, applied when IPMA precipitation probability >= threshold (%)
VIS_RAIN_PROB_THRESHOLD = 70
VIS_RAIN_PENALTY_RIA = 0.75    # sites next to the Ria Formosa inlets
VIS_RAIN_PENALTY_COAST = 0.85  # rest of the coast (ribeiras, Salgados lagoon)
