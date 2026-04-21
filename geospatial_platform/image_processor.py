import numpy as np
import sys
sys.path.append("/kaggle/working")

from geospatial_platform.context import InputContext

BAND_CONFIG = {
    "sentinel2":      {"red": 3,  "green": 2, "blue": 1, "nir": 7,  "swir": 10},
    "sentinel2_6band":{"red": 2,  "green": 1, "blue": 0, "nir": 3,  "swir": 4},
    "landsat8":       {"red": 3,  "green": 2, "blue": 1, "nir": 4,  "swir": 5},
    "generic4":       {"red": 0,  "green": 1, "blue": 2, "nir": 3,  "swir": None},
    "rgb":            {"red": 0,  "green": 1, "blue": 2, "nir": None,"swir": None},
}


def detect_sensor(n_bands: int) -> str:
    if n_bands == 3:   return "rgb"
    elif n_bands == 4: return "generic4"
    elif n_bands == 6: return "sentinel2_6band"
    elif n_bands == 7: return "landsat8"
    elif n_bands >= 13: return "sentinel2"
    else: return "generic4"


def normalize_band(band: np.ndarray) -> np.ndarray:
    min_val, max_val = np.nanmin(band), np.nanmax(band)
    if max_val - min_val == 0:
        return np.zeros_like(band, dtype=np.float32)
    return ((band - min_val) / (max_val - min_val)).astype(np.float32)

NODATA_THRESHOLD = -0.15  # GEE Landsat scaled nodata ≈ -0.2

def mask_nodata(array: np.ndarray) -> np.ndarray:
    """Replace nodata pixels (-0.2 from GEE scaling) with NaN."""
    masked = array.astype(np.float32).copy()
    masked[masked <= NODATA_THRESHOLD] = np.nan
    return masked


def safe_index(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute (a-b)/(a+b) safely, ignoring NaN and zero denominators."""
    denom  = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(
            (np.abs(denom) < 1e-10) | np.isnan(denom),
            np.nan,
            (a - b) / denom
        )
    return np.clip(result, -1.0, 1.0).astype(np.float32)


def compute_ndvi(array, config):
    if config["nir"] is None: return None
    nir = mask_nodata(normalize_band(array[config["nir"]], prescaled=True))
    red = mask_nodata(normalize_band(array[config["red"]], prescaled=True))
    return safe_index(nir, red)


def compute_ndwi(array, config):
    if config["nir"] is None: return None
    green = mask_nodata(normalize_band(array[config["green"]], prescaled=True))
    nir   = mask_nodata(normalize_band(array[config["nir"]],   prescaled=True))
    return safe_index(green, nir)


def compute_ndbi(array, config):
    if config["swir"] is None or config["nir"] is None: return None
    swir = mask_nodata(normalize_band(array[config["swir"]], prescaled=True))
    nir  = mask_nodata(normalize_band(array[config["nir"]],  prescaled=True))
    return safe_index(swir, nir)

def compute_vegetation_breakdown(ndvi: np.ndarray, total_pixels: int) -> dict:
    valid = ndvi[~np.isnan(ndvi)]
    if len(valid) == 0:
        return {"sparse_pct": 0, "moderate_pct": 0, "dense_pct": 0}
    sparse   = ((valid > 0.1)  & (valid <= 0.25)).sum()
    moderate = ((valid > 0.25) & (valid <= 0.45)).sum()
    dense    = (valid > 0.45).sum()
    return {
        "sparse_pct":   round(float(sparse)   / total_pixels * 100, 2),
        "moderate_pct": round(float(moderate) / total_pixels * 100, 2),
        "dense_pct":    round(float(dense)    / total_pixels * 100, 2),
    }


def detect_flood(ndwi: np.ndarray, total_pixels: int) -> float:
    valid = ndwi[~np.isnan(ndwi)]
    if len(valid) == 0:
        return 0.0
    water_pixels = (valid > 0.3).sum()
    return round(float(water_pixels) / total_pixels, 4)


def detect_flood(ndwi: np.ndarray, total_pixels: int) -> float:
    """Return fraction of pixels with NDWI > 0.3 (water/flood signal)."""
    water_pixels = (ndwi > 0.3).sum()
    return round(float(water_pixels) / total_pixels, 4)


def generate_feature_maps(array: np.ndarray, config: dict) -> dict:
    maps = {}
    ndvi = compute_ndvi(array, config)
    ndwi = compute_ndwi(array, config)
    ndbi = compute_ndbi(array, config)

    if ndvi is not None:
        valid_ndvi = np.where(np.isnan(ndvi), 0, ndvi)
        veg_mean   = float(np.nanmean(ndvi)) if not np.all(np.isnan(ndvi)) else 0
        veg_std    = float(np.nanstd(ndvi))  if not np.all(np.isnan(ndvi)) else 0
        veg_threshold = max(0.1, veg_mean + 0.3 * veg_std)
        maps["vegetation_mask"] = (valid_ndvi > veg_threshold).astype(np.uint8)
        print(f"  NDVI adaptive threshold: {veg_threshold:.3f}")

    if ndwi is not None:
        valid_ndwi  = np.where(np.isnan(ndwi), 0, ndwi)
        water_mean  = float(np.nanmean(ndwi)) if not np.all(np.isnan(ndwi)) else 0
        water_std   = float(np.nanstd(ndwi))  if not np.all(np.isnan(ndwi)) else 0
        water_threshold = max(0.0, water_mean + 0.5 * water_std)
        maps["water_mask"] = (valid_ndwi > water_threshold).astype(np.uint8)
        print(f"  NDWI adaptive threshold: {water_threshold:.3f}")

    if ndbi is not None:
        valid_ndbi  = np.where(np.isnan(ndbi), 0, ndbi)
        urban_mean  = float(np.nanmean(ndbi)) if not np.all(np.isnan(ndbi)) else 0
        urban_std   = float(np.nanstd(ndbi))  if not np.all(np.isnan(ndbi)) else 0
        urban_threshold = max(0.05, urban_mean + 0.5 * urban_std)
        maps["urban_mask"] = (valid_ndbi > urban_threshold).astype(np.uint8)
        print(f"  NDBI adaptive threshold: {urban_threshold:.3f}")

    return maps


def compute_cover_percentages(feature_maps: dict, total_pixels: int) -> dict:
    """Mutually exclusive land cover — water > vegetation > urban > barren."""
    side = int(total_pixels ** 0.5)
    classified = np.zeros((side, side), dtype=np.uint8)

    priority = ["water_mask", "vegetation_mask", "urban_mask"]
    labels   = {"water_mask": "water", "vegetation_mask": "vegetation", "urban_mask": "urban"}
    codes    = {"water_mask": 1, "vegetation_mask": 2, "urban_mask": 3}

    for mask_name in priority:
        if mask_name in feature_maps:
            mask = feature_maps[mask_name]
            if mask.shape == classified.shape:
                classified[mask.astype(bool) & (classified == 0)] = codes[mask_name]

    percentages = {}
    for mask_name in priority:
        pct = round(float((classified == codes[mask_name]).sum()) / total_pixels * 100, 2)
        percentages[labels[mask_name]] = pct

    percentages["barren"] = round(float((classified == 0).sum()) / total_pixels * 100, 2)
    return percentages


def compute_aridity_index(context: InputContext) -> float | None:
    """
    UNEP aridity index = P / PET
    P   = annual precipitation (mm)
    PET = potential evapotranspiration (mm) ≈ 58.93 * mean_temp (Thornthwaite simplified)
    Scale: <0.05=hyper-arid, 0.05-0.2=arid, 0.2-0.5=semi-arid,
           0.5-0.65=dry sub-humid, >0.65=humid
    """
    if context.csv_df is None:
        return None
    df = context.csv_df
    temp_cols = [c for c in df.columns if "temp" in c.lower() or "t2m" in c.lower()]
    rain_cols = [c for c in df.columns if "rain" in c.lower() or "prec" in c.lower()]
    if not temp_cols or not rain_cols:
        return None

    mean_temp  = float(df[temp_cols[0]].mean())
    total_rain = float(df[rain_cols[0]].sum())

    # Thornthwaite simplified PET (mm/year)
    pet = max(1.0, 58.93 * mean_temp)
    ai  = total_rain / pet
    return round(ai, 4)


def classify_aridity(index: float) -> str:
    if index < 0.05:  return "Hyper-arid"
    elif index < 0.2: return "Arid"
    elif index < 0.5: return "Semi-arid"
    elif index < 0.65:return "Dry sub-humid"
    else:             return "Humid"


def compute_seasonality(context: InputContext) -> str:
    """Detect rainfall seasonality pattern."""
    if context.csv_df is None:
        return "unknown"
    df = context.csv_df
    rain_cols = [c for c in df.columns if "rain" in c.lower() or "prec" in c.lower()]
    if not rain_cols:
        return "unknown"
    series = df[rain_cols[0]].dropna()
    if len(series) < 3 or series.mean() == 0:
        return "unknown"
    std_ratio = series.std() / series.mean()
    if std_ratio > 1.0:   return "strongly seasonal"
    elif std_ratio > 0.5: return "moderately seasonal"
    else:                 return "relatively uniform"


def classify_ecosystem(land_cover: dict, aridity_index: float = None,
                        ndvi_mean: float = None) -> str:
    """
    Classify ecosystem type from land cover ratios and indices.
    Returns a human-readable ecosystem label.
    """
    water      = land_cover.get("water", 0)
    vegetation = land_cover.get("vegetation", 0)
    urban      = land_cover.get("urban", 0)
    barren     = land_cover.get("barren", 0)

    # Water-dominated
    if water > 50:
        return "Aquatic / Wetland"

    # Urban-dominated
    if urban > 40:
        return "Urban / Built-up area"

    # Highly vegetated
    if vegetation > 60:
        if ndvi_mean and ndvi_mean > 0.5:
            return "Dense forest / Tropical vegetation"
        return "Agricultural land / Grassland"

    # Mixed with aridity context
    if aridity_index is not None:
        if aridity_index < 0.05:
            return "Hyper-arid desert"
        elif aridity_index < 0.2:
            if barren > 40:
                return "Arid shrubland / Desert"
            return "Arid rangeland"
        elif aridity_index < 0.5:
            if vegetation > 25:
                return "Semi-arid savanna / Mediterranean scrubland"
            return "Semi-arid barren land"
        elif aridity_index < 0.65:
            return "Dry sub-humid mixed land"
        else:
            if vegetation > 40:
                return "Humid forest / Dense vegetation"
            return "Humid mixed landscape"

    # Fallback from land cover only
    if barren > 50:
        return "Barren / Sparsely vegetated land"
    if vegetation > 30 and urban > 15:
        return "Peri-urban mixed landscape"
    if vegetation > 40:
        return "Mixed agricultural / Natural vegetation"

    return "Mixed / Unclassified landscape"
                            
def process_image(context: InputContext, sensor: str = None) -> InputContext:
    print("=== Image Processor ===")

    array   = context.image_array
    n_bands = context.n_bands
    sensor  = sensor or detect_sensor(n_bands)
    config  = BAND_CONFIG[sensor]

    print(f"  Sensor type : {sensor}")
    print(f"  Band config : {config}")

    normalized = np.stack([normalize_band(array[i]) for i in range(n_bands)])
    context.image_meta["normalized"] = True

    context.ndvi = compute_ndvi(array, config)
    context.ndwi = compute_ndwi(array, config)
    context.ndbi = compute_ndbi(array, config)

    for name, val in [("NDVI", context.ndvi),
                  ("NDWI", context.ndwi),
                  ("NDBI", context.ndbi)]:
    if val is not None:
        valid = val[~np.isnan(val)]
        if len(valid) > 0:
            print(f"  {name}: min={valid.min():.3f}, "
                  f"mean={valid.mean():.3f}, max={valid.max():.3f}, "
                  f"valid_pixels={len(valid)}")
        else:
            print(f"  {name}: all NaN — possible nodata issue")
    else:
        print(f"  {name}: skipped")

    total_pixels = array.shape[1] * array.shape[2]
    feature_maps = generate_feature_maps(array, config)
    coverage     = compute_cover_percentages(feature_maps, total_pixels)

    if coverage:
        print(f"  Coverage: {coverage}")
        context.land_cover = coverage
    else:
        context.land_cover = {}

    # Vegetation breakdown
    if context.ndvi is not None:
        context.vegetation_breakdown = compute_vegetation_breakdown(
            context.ndvi, total_pixels)
        print(f"  Vegetation breakdown: {context.vegetation_breakdown}")

    # Flood detection
    if context.ndwi is not None:
        context.water_ratio = detect_flood(context.ndwi, total_pixels)
        print(f"  Water/flood ratio: {context.water_ratio:.3f}")

    # Aridity index
    context.aridity_index = compute_aridity_index(context)
    if context.aridity_index:
        print(f"  Aridity index: {context.aridity_index:.3f}")

    context.image_array = normalized
    # Ecosystem classification
    ndvi_mean_val = float(context.ndvi.mean()) if context.ndvi is not None else None
    context.image_meta["ecosystem"] = classify_ecosystem(
        context.land_cover,
        context.aridity_index,
        ndvi_mean_val
    )
    print(f"  Ecosystem: {context.image_meta['ecosystem']}")

    print("=== Image processing complete ===\n")
    return context
