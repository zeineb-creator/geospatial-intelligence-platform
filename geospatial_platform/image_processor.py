"""
image_processor.py — Sensor-agnostic land cover classification
==============================================================
Approach: Multi-index ensemble with soft voting.

Instead of relying on a single index (NDBI) with a fixed threshold,
this module:
  1. Detects sensor type and maps bands correctly
  2. Computes every spectral index available for that sensor
  3. Scores each pixel across all indices using soft membership functions
  4. Votes across indices to produce a final classification

This is sensor-agnostic because normalised indices (NDVI, BSI, MNDWI, etc.)
cancel out radiometric differences between Sentinel-2, Landsat 5/7/8/9, and
generic multispectral sensors.
"""

import numpy as np
import sys
sys.path.append("/kaggle/working")

from geospatial_platform.context import InputContext


# ══════════════════════════════════════════════════════════════════════════════
# SENSOR BAND CONFIGURATIONS
# Keys: blue, green, red, nir, swir1, swir2 (0-indexed band positions)
# None = band not available for that sensor
# ══════════════════════════════════════════════════════════════════════════════

BAND_CONFIG = {
    # Sentinel-2 L2A — 13 bands (10m+20m+60m resampled)
    "sentinel2": {
        "blue": 1, "green": 2, "red": 3, "nir": 7,
        "swir1": 10, "swir2": 11,
    },
    # Sentinel-2 L2A — 6-band export (Blue/Green/Red/NIR/SWIR1/SWIR2)
    "sentinel2_6band": {
        "blue": 0, "green": 1, "red": 2, "nir": 3,
        "swir1": 4, "swir2": 5,
    },
    # Landsat 8/9 OLI (bands 1-7 exported as 0-indexed)
    "landsat8": {
        "blue": 1, "green": 2, "red": 3, "nir": 4,
        "swir1": 5, "swir2": 6,
    },
    # Landsat 5/7 TM (bands 1-7, band 6 thermal omitted)
    "landsat5": {
        "blue": 0, "green": 1, "red": 2, "nir": 3,
        "swir1": 4, "swir2": 5,
    },
    # Landsat 7 ETM+ (same layout as Landsat 5 for optical)
    "landsat7": {
        "blue": 0, "green": 1, "red": 2, "nir": 3,
        "swir1": 4, "swir2": 5,
    },
    # Generic 4-band (B/G/R/NIR — no SWIR)
    "generic4": {
        "blue": 0, "green": 1, "red": 2, "nir": 3,
        "swir1": None, "swir2": None,
    },
    # RGB only — no NIR, no SWIR
    "rgb": {
        "blue": 2, "green": 1, "red": 0,
        "nir": None, "swir1": None, "swir2": None,
    },
}

NODATA_THRESHOLD = -0.25


# ══════════════════════════════════════════════════════════════════════════════
# SENSOR DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_sensor(n_bands: int, meta: dict = None) -> str:
    """
    Infer sensor from band count.
    If meta contains a 'sensor' key it takes priority.
    """
    if meta and meta.get("sensor"):
        return meta["sensor"]
    if n_bands == 3:   return "rgb"
    if n_bands == 4:   return "generic4"
    if n_bands == 6:   return "sentinel2_6band"
    if n_bands == 7:   return "landsat8"
    if n_bands >= 13:  return "sentinel2"
    return "generic4"


def is_prescaled(array: np.ndarray) -> bool:
    """True if values are already in reflectance [0, 1]."""
    valid = array[~np.isnan(array)]
    valid = valid[valid > -999]
    if len(valid) == 0:
        return False
    return float(valid.max()) <= 1.5 and float(valid.min()) >= -0.5


def get_band(array: np.ndarray, config: dict, name: str) -> np.ndarray | None:
    """Safely extract a band by logical name. Returns None if not available."""
    idx = config.get(name)
    if idx is None or idx >= array.shape[0]:
        return None
    b = array[idx].astype(np.float32).copy()
    b[b <= NODATA_THRESHOLD] = np.nan
    return b


def normalize_to_reflectance(band: np.ndarray, prescaled: bool) -> np.ndarray:
    """Scale to [0, 1] if not already reflectance."""
    if prescaled:
        return np.clip(band, 0.0, 1.0)
    mn, mx = np.nanmin(band), np.nanmax(band)
    if mx - mn < 1e-6:
        return np.zeros_like(band)
    return np.clip((band - mn) / (mx - mn), 0.0, 1.0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# SPECTRAL INDEX COMPUTATION (all sensor-agnostic)
# ══════════════════════════════════════════════════════════════════════════════

def _ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b), clipped to [-1, 1], NaN where denominator ~0."""
    denom = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(
            (np.abs(denom) < 1e-8) | np.isnan(denom),
            np.nan,
            (a - b) / denom,
        )
    return np.clip(result, -1.0, 1.0).astype(np.float32)


def compute_ndvi(red, nir):
    """Normalised Difference Vegetation Index — vegetation signal."""
    if red is None or nir is None:
        return None
    return _ratio(nir, red)


def compute_ndwi(green, nir):
    """Normalised Difference Water Index (McFeeters) — open water."""
    if green is None or nir is None:
        return None
    return _ratio(green, nir)


def compute_mndwi(green, swir1):
    """Modified NDWI (Xu 2006) — better water/urban separation than NDWI."""
    if green is None or swir1 is None:
        return None
    return _ratio(green, swir1)


def compute_ndbi(swir1, nir):
    """Normalised Difference Built-up Index."""
    if swir1 is None or nir is None:
        return None
    return _ratio(swir1, nir)


def compute_bsi(blue, red, nir, swir1):
    """
    Bare Soil Index — separates bare/urban from vegetation.
    BSI = ((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))
    """
    if any(b is None for b in [blue, red, nir, swir1]):
        return None
    num = (swir1 + red) - (nir + blue)
    den = (swir1 + red) + (nir + blue)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(np.abs(den) < 1e-8, np.nan, num / den)
    return np.clip(result, -1.0, 1.0).astype(np.float32)


def compute_ui(swir2, nir):
    """Urban Index — SWIR2/NIR ratio, high for impervious surfaces."""
    if swir2 is None or nir is None:
        return None
    return _ratio(swir2, nir)


def compute_evi2(red, nir):
    """Two-band EVI — robust vegetation index for bright surfaces."""
    if red is None or nir is None:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        result = 2.5 * (nir - red) / (nir + 2.4 * red + 1.0)
    return np.clip(result, -1.0, 1.0).astype(np.float32)


def compute_savi(red, nir, L=0.5):
    """Soil-Adjusted Vegetation Index — reduces soil background effect."""
    if red is None or nir is None:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        result = ((nir - red) / (nir + red + L)) * (1 + L)
    return np.clip(result, -1.0, 1.0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# SOFT MEMBERSHIP FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
# Each function returns a [0, 1] "probability of belonging to class X" per pixel.
# This is the core of the sensor-agnostic approach: instead of hard thresholds,
# we use smooth sigmoid-based transitions that are robust to sensor radiometry.

def _sigmoid(x: np.ndarray, center: float, slope: float) -> np.ndarray:
    """Logistic function: 0 below center, 1 above center."""
    return (1.0 / (1.0 + np.exp(-slope * (x - center)))).astype(np.float32)


def _inv_sigmoid(x: np.ndarray, center: float, slope: float) -> np.ndarray:
    return 1.0 - _sigmoid(x, center, slope)


def _bell(x: np.ndarray, center: float, width: float) -> np.ndarray:
    """Bell curve peaking at center."""
    return np.exp(-0.5 * ((x - center) / (width + 1e-8)) ** 2).astype(np.float32)


def _safe(arr):
    """Replace NaN with 0 for membership calculation."""
    return np.nan_to_num(arr, nan=0.0)


def water_score(ndwi, mndwi, ndvi) -> np.ndarray:
    """High score = likely water."""
    scores = []
    if ndwi is not None:
        scores.append(_sigmoid(_safe(ndwi), center=0.0, slope=20))
    if mndwi is not None:
        # MNDWI is more reliable — give it higher weight
        scores.append(_sigmoid(_safe(mndwi), center=0.05, slope=20) * 1.5)
    if ndvi is not None:
        # Water has very low NDVI — negative correlation
        scores.append(_inv_sigmoid(_safe(ndvi), center=0.1, slope=20) * 0.5)
    if not scores:
        return np.zeros(1)
    total_weight = 1.0 + (1.5 if mndwi is not None else 0) + (0.5 if ndvi is not None else 0)
    return np.clip(np.sum(scores, axis=0) / total_weight, 0, 1)


def vegetation_score(ndvi, evi2, savi, mndwi) -> np.ndarray:
    """High score = likely vegetation."""
    scores = []
    if ndvi is not None:
        scores.append(_sigmoid(_safe(ndvi), center=0.15, slope=25))
    if evi2 is not None:
        scores.append(_sigmoid(_safe(evi2), center=0.1, slope=25))
    if savi is not None:
        scores.append(_sigmoid(_safe(savi), center=0.12, slope=25) * 0.8)
    if mndwi is not None:
        # Vegetation has negative MNDWI — penalise high MNDWI
        scores.append(_inv_sigmoid(_safe(mndwi), center=0.0, slope=15) * 0.5)
    if not scores:
        return np.zeros(1)
    total_weight = (1 + (1 if evi2 is not None else 0) +
                    (0.8 if savi is not None else 0) +
                    (0.5 if mndwi is not None else 0))
    return np.clip(np.sum(scores, axis=0) / total_weight, 0, 1)


def urban_score(ndbi, bsi, ui, mndwi, ndvi) -> np.ndarray:
    """
    High score = likely urban/built-up.
    Uses BSI and UI as primary signals — more reliable than NDBI alone
    because BSI uses 4 bands and UI uses SWIR2 (very sensitive to impervious surfaces).
    """
    scores = []
    weights = []

    if ndbi is not None:
        scores.append(_sigmoid(_safe(ndbi), center=0.0, slope=15))
        weights.append(1.0)

    if bsi is not None:
        # BSI is the best single indicator for bare/urban across sensors
        scores.append(_sigmoid(_safe(bsi), center=0.0, slope=20))
        weights.append(2.0)  # double weight

    if ui is not None:
        scores.append(_sigmoid(_safe(ui), center=0.0, slope=15))
        weights.append(1.5)

    if mndwi is not None:
        # Urban has negative MNDWI (unlike water which is positive)
        scores.append(_inv_sigmoid(_safe(mndwi), center=0.0, slope=15) * 0.8)
        weights.append(0.8)

    if ndvi is not None:
        # Urban has low NDVI
        scores.append(_inv_sigmoid(_safe(ndvi), center=0.2, slope=20) * 0.6)
        weights.append(0.6)

    if not scores:
        return np.zeros(1)

    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    return np.clip(weighted_sum / sum(weights), 0, 1)


def barren_score(ndvi, bsi, mndwi) -> np.ndarray:
    """High score = likely bare soil / barren."""
    scores = []
    if ndvi is not None:
        # Barren = low NDVI
        scores.append(_inv_sigmoid(_safe(ndvi), center=0.1, slope=20))
    if bsi is not None:
        # BSI peaks for bare soil — bell around 0.1
        scores.append(_bell(_safe(bsi), center=0.1, width=0.2))
    if mndwi is not None:
        # Barren is negative MNDWI (dry) but not as negative as urban
        scores.append(_bell(_safe(mndwi), center=-0.15, width=0.2))
    if not scores:
        return np.zeros(1)
    return np.clip(np.mean(scores, axis=0), 0, 1)


# ══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

def classify_ensemble(indices: dict) -> np.ndarray:
    """
    Soft-vote classification returning a (H, W) label array.
    Labels: 1=water, 2=vegetation, 3=urban, 4=barren
    """
    ndvi   = indices.get("ndvi")
    ndwi   = indices.get("ndwi")
    mndwi  = indices.get("mndwi")
    ndbi   = indices.get("ndbi")
    bsi    = indices.get("bsi")
    ui     = indices.get("ui")
    evi2   = indices.get("evi2")
    savi   = indices.get("savi")

    # Compute soft membership for each class
    w_water = water_score(ndwi, mndwi, ndvi)
    w_veg   = vegetation_score(ndvi, evi2, savi, mndwi)
    w_urban = urban_score(ndbi, bsi, ui, mndwi, ndvi)
    w_bare  = barren_score(ndvi, bsi, mndwi)

    # Stack scores and take argmax
    scores = np.stack([w_water, w_veg, w_urban, w_bare], axis=0)
    labels = np.argmax(scores, axis=0).astype(np.uint8) + 1  # 1-indexed
    # Label map: 1=water, 2=vegetation, 3=urban, 4=barren

    return labels, scores


def compute_cover_percentages(labels: np.ndarray) -> dict:
    """Convert label map to percentage coverage dict."""
    total = labels.size
    return {
        "water":      round(float((labels == 1).sum()) / total * 100, 2),
        "vegetation": round(float((labels == 2).sum()) / total * 100, 2),
        "urban":      round(float((labels == 3).sum()) / total * 100, 2),
        "barren":     round(float((labels == 4).sum()) / total * 100, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# AUXILIARY METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_aridity_index(context: InputContext) -> float | None:
    if context.csv_df is None:
        return None
    df = context.csv_df
    temp_cols = [c for c in df.columns if "temp" in c.lower() or "t2m" in c.lower()]
    rain_cols = [c for c in df.columns if "rain" in c.lower() or "prec" in c.lower()]
    if not temp_cols or not rain_cols:
        return None
    mean_temp  = float(df[temp_cols[0]].mean())
    total_rain = float(df[rain_cols[0]].sum())
    pet = max(1.0, 58.93 * mean_temp)
    return round(total_rain / (pet + 1e-6), 4)


def classify_ecosystem(land_cover: dict, aridity_index=None, ndvi_mean=None) -> str:
    water = land_cover.get("water", 0)
    veg   = land_cover.get("vegetation", 0)
    urban = land_cover.get("urban", 0)
    bare  = land_cover.get("barren", 0)

    if water > 50:
        return "Aquatic / Wetland"
    if water > 15 and veg > 5:
        return "Mediterranean coastal mixed landscape"
    if urban > 40:
        return "Urban / Built-up area"
    if veg > 60:
        return "Dense forest / Tropical vegetation" if (ndvi_mean and ndvi_mean > 0.5) else "Agricultural / Grassland"
    if veg > 20 and urban > 10:
        return "Peri-urban mixed landscape"
    if aridity_index is not None:
        if aridity_index < 0.05:  return "Hyper-arid desert"
        if aridity_index < 0.2:   return "Arid shrubland / Desert" if veg < 15 else "Semi-arid Mediterranean scrubland"
        if aridity_index < 0.5:   return "Semi-arid savanna / Mediterranean scrubland" if veg > 25 else "Semi-arid barren land"
        if aridity_index < 0.65:  return "Dry sub-humid mixed land"
        return "Humid forest / Dense vegetation" if veg > 40 else "Humid mixed landscape"
    if bare > 60:
        return "Barren / Sparsely vegetated land"
    return "Mixed / Unclassified landscape"


def compute_vegetation_breakdown(ndvi: np.ndarray, total_pixels: int) -> dict:
    valid = ndvi[~np.isnan(ndvi)]
    if len(valid) == 0:
        return {"sparse_pct": 0.0, "moderate_pct": 0.0, "dense_pct": 0.0}
    return {
        "sparse_pct":   round(float(((valid > 0.1)  & (valid <= 0.25)).sum()) / total_pixels * 100, 2),
        "moderate_pct": round(float(((valid > 0.25) & (valid <= 0.45)).sum()) / total_pixels * 100, 2),
        "dense_pct":    round(float((valid > 0.45).sum())                      / total_pixels * 100, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def process_image(context: InputContext, sensor: str = None) -> InputContext:
    print("=== Image Processor (multi-index ensemble) ===")

    array   = context.image_array
    n_bands = context.n_bands
    sensor  = sensor or detect_sensor(n_bands, context.image_meta)
    config  = BAND_CONFIG[sensor]

    print(f"  Sensor       : {sensor}")
    print(f"  Array shape  : {array.shape} | range [{array.min():.3f}, {array.max():.3f}]")
    print(f"  Band config  : {config}")

    prescaled = is_prescaled(array)
    print(f"  Pre-scaled   : {prescaled}")

    # ── Extract and normalise all available bands ─────────────────────────────
    bands = {}
    for name in ["blue", "green", "red", "nir", "swir1", "swir2"]:
        raw = get_band(array, config, name)
        if raw is not None:
            bands[name] = normalize_to_reflectance(raw, prescaled)
        else:
            bands[name] = None

    available = [k for k, v in bands.items() if v is not None]
    print(f"  Available bands: {available}")

    # ── Compute all spectral indices ──────────────────────────────────────────
    indices = {}
    indices["ndvi"]  = compute_ndvi(bands["red"],   bands["nir"])
    indices["ndwi"]  = compute_ndwi(bands["green"], bands["nir"])
    indices["mndwi"] = compute_mndwi(bands["green"], bands["swir1"])
    indices["ndbi"]  = compute_ndbi(bands["swir1"], bands["nir"])
    indices["bsi"]   = compute_bsi(bands["blue"], bands["red"], bands["nir"], bands["swir1"])
    indices["ui"]    = compute_ui(bands["swir2"], bands["nir"])
    indices["evi2"]  = compute_evi2(bands["red"], bands["nir"])
    indices["savi"]  = compute_savi(bands["red"], bands["nir"])

    computed = [k for k, v in indices.items() if v is not None]
    print(f"  Computed indices: {computed}")

    for name, arr in indices.items():
        if arr is not None:
            valid = arr[~np.isnan(arr)]
            if len(valid) > 0:
                print(f"  {name.upper():6s}: mean={valid.mean():.3f} min={valid.min():.3f} max={valid.max():.3f}")

    # ── Store primary indices on context ──────────────────────────────────────
    context.ndvi     = indices["ndvi"]
    context.ndwi     = indices["ndwi"]
    context.ndbi     = indices["ndbi"]
    context.ndvi_map = indices["ndvi"]
    context.ndwi_map = indices["ndwi"]
    context.ndbi_map = indices["ndbi"]

    if context.ndvi is not None:
        context.ndvi_mean = float(np.nanmean(context.ndvi))
    if context.ndwi is not None:
        context.ndwi_mean = float(np.nanmean(context.ndwi))
    if context.ndbi is not None:
        context.ndbi_mean = float(np.nanmean(context.ndbi))

    # ── Ensemble classification ───────────────────────────────────────────────
    labels, score_stack = classify_ensemble(indices)
    land_cover = compute_cover_percentages(labels)
    print(f"  Land cover   : {land_cover}")

    context.land_cover = land_cover

    # ── Derived metrics ───────────────────────────────────────────────────────
    context.aridity_index = compute_aridity_index(context)
    if context.aridity_index:
        print(f"  Aridity index: {context.aridity_index:.3f}")

    ndvi_mean_val = context.ndvi_mean
    context.image_meta["ecosystem"] = classify_ecosystem(
        land_cover, context.aridity_index, ndvi_mean_val
    )
    context.ecosystem = context.image_meta["ecosystem"]
    context.region    = context.image_meta.get("region_name", "Unknown region")
    print(f"  Ecosystem    : {context.ecosystem}")

    # Store normalised array for ViT
    norm_stack = []
    for i in range(min(n_bands, array.shape[0])):
        raw = array[i].astype(np.float32).copy()
        raw[raw <= NODATA_THRESHOLD] = np.nan
        norm_stack.append(normalize_to_reflectance(np.nan_to_num(raw, nan=0.0), prescaled))
    context.image_array = np.stack(norm_stack)
    context.image_meta["normalized"] = True
    context.image_meta["prescaled"]  = prescaled
    context.image_meta["sensor"]     = sensor
    context.image_meta["indices_computed"] = computed

    print("=== Image processing complete ===\n")
    return context
