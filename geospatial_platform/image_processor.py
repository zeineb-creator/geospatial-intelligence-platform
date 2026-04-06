
import numpy as np
import sys
sys.path.append("/kaggle/working")

from geospatial_platform.context import InputContext


# --- Band index mapping by sensor type ---
# Sentinel-2: B4=Red, B3=Green, B2=Blue, B8=NIR, B11=SWIR
# Landsat 8:  B4=Red, B3=Green, B2=Blue, B5=NIR, B6=SWIR
# Generic 4-band: 0=Red, 1=Green, 2=Blue, 3=NIR

BAND_CONFIG = {
    "sentinel2": {"red": 3, "green": 2, "blue": 1, "nir": 7, "swir": 10},
    "sentinel2_6band": {"red": 2, "green": 1, "blue": 0, "nir": 3, "swir": 4},
    "landsat8":  {"red": 3, "green": 2, "blue": 1, "nir": 4,  "swir": 5},
    "generic4":  {"red": 0, "green": 1, "blue": 2, "nir": 3,  "swir": None},
    "rgb":       {"red": 0, "green": 1, "blue": 2, "nir": None, "swir": None},
}


def detect_sensor(n_bands: int) -> str:
    """Guess sensor type from band count."""
    if n_bands == 3:
        return "rgb"
    elif n_bands == 4:
        return "generic4"
    elif n_bands == 13:
        return "sentinel2"
    elif n_bands >= 7:
        return "landsat8"
    else:
        return "rgb"


def normalize_band(band: np.ndarray) -> np.ndarray:
    """Normalize a single band to [0, 1] range."""
    min_val = np.nanmin(band)
    max_val = np.nanmax(band)
    if max_val - min_val == 0:
        return np.zeros_like(band, dtype=np.float32)
    return ((band - min_val) / (max_val - min_val)).astype(np.float32)


def safe_index(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute (a - b) / (a + b) safely, avoiding division by zero.
    Returns values in [-1, 1].
    """
    denom = a + b
    result = np.where(denom == 0, 0.0, (a - b) / denom)
    return np.clip(result, -1.0, 1.0).astype(np.float32)


def compute_ndvi(array: np.ndarray, config: dict) -> np.ndarray | None:
    """NDVI = (NIR - Red) / (NIR + Red)"""
    if config["nir"] is None:
        return None
    nir = normalize_band(array[config["nir"]])
    red = normalize_band(array[config["red"]])
    return safe_index(nir, red)


def compute_ndwi(array: np.ndarray, config: dict) -> np.ndarray | None:
    """NDWI = (Green - NIR) / (Green + NIR)"""
    if config["nir"] is None:
        return None
    green = normalize_band(array[config["green"]])
    nir   = normalize_band(array[config["nir"]])
    return safe_index(green, nir)


def compute_ndbi(array: np.ndarray, config: dict) -> np.ndarray | None:
    """NDBI = (SWIR - NIR) / (SWIR + NIR)"""
    if config["swir"] is None or config["nir"] is None:
        return None
    swir = normalize_band(array[config["swir"]])
    nir  = normalize_band(array[config["nir"]])
    return safe_index(swir, nir)


def generate_feature_maps(array: np.ndarray, config: dict) -> dict:
    """
    Apply thresholds to indices to produce binary masks.
    Returns dict of masks: vegetation, water, urban.
    """
    maps = {}

    ndvi = compute_ndvi(array, config)
    ndwi = compute_ndwi(array, config)
    ndbi = compute_ndbi(array, config)

    if ndvi is not None:
        maps["vegetation_mask"] = (ndvi > 0.2).astype(np.uint8)
    if ndwi is not None:
        maps["water_mask"] = (ndwi > 0.0).astype(np.uint8)
    if ndbi is not None:
        maps["urban_mask"] = (ndbi > 0.0).astype(np.uint8)

    return maps


def compute_cover_percentages(feature_maps: dict, total_pixels: int) -> dict:
    """
    Convert binary masks to percentage coverage.
    """
    percentages = {}
    for name, mask in feature_maps.items():
        label = name.replace("_mask", "")
        pct = round(float(mask.sum()) / total_pixels * 100, 2)
        percentages[label] = pct
    return percentages


def process_image(context: InputContext, sensor: str = None) -> InputContext:
    """
    Main entry point for image processing.
    Mutates and returns the InputContext with computed indices + feature maps.
    """
    print("=== Image Processor ===")

    array   = context.image_array
    n_bands = context.n_bands

    # Auto-detect sensor if not specified
    sensor = sensor or detect_sensor(n_bands)
    config = BAND_CONFIG[sensor]
    print(f"  Sensor type : {sensor}")
    print(f"  Band config : {config}")

    # Normalize full image
    normalized = np.stack([normalize_band(array[i]) for i in range(n_bands)])
    context.image_meta["normalized"] = True

    # Compute indices
    context.ndvi = compute_ndvi(array, config)
    context.ndwi = compute_ndwi(array, config)
    context.ndbi = compute_ndbi(array, config)

    # Report
    for name, val in [("NDVI", context.ndvi), ("NDWI", context.ndwi), ("NDBI", context.ndbi)]:
        if val is not None:
            print(f"  {name}: min={val.min():.3f}, mean={val.mean():.3f}, max={val.max():.3f}")
        else:
            print(f"  {name}: skipped (missing required bands)")

    # Feature maps + coverage
    total_pixels = array.shape[1] * array.shape[2]
    feature_maps = generate_feature_maps(array, config)
    coverage = compute_cover_percentages(feature_maps, total_pixels)

    if coverage:
        print(f"  Coverage estimates: {coverage}")
        context.land_cover = coverage
    else:
        print("  Coverage: not computable (RGB only)")
        context.land_cover = {}

    # Store normalized array back
    context.image_array = normalized

    print("=== Image processing complete ===\n")
    return context
