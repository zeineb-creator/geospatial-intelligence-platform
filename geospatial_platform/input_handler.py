
import numpy as np
import pandas as pd
from PIL import Image
from dataclasses import dataclass
import os
import sys

from geospatial_platform.context import InputContext

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

def get_region_name(meta: dict) -> str:
    """
    Attempt reverse geocoding from image bounds using Nominatim.
    Falls back gracefully if no internet or coordinates unavailable.
    """
    try:
        import urllib.request
        import json

        crs = str(meta.get("crs", ""))
        if "4326" not in crs:
            return "Unknown region"

        # Get center coordinates from transform
        transform = meta.get("transform", None)
        width     = meta.get("width", 0)
        height    = meta.get("height", 0)

        if transform is None:
            return "Unknown region"

        # Compute center lat/lon
        import rasterio.transform as rt
        lon, lat = rt.xy(transform, height // 2, width // 2)

        url = (
            f"https://nominatim.openstreetmap.org/reverse"
            f"?lat={lat}&lon={lon}&format=json"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "GeoAI-Platform/1.0"})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read())

        address = data.get("address", {})
        parts = []
        for key in ["city", "town", "village", "county", "state", "country"]:
            if key in address and address[key]:
                parts.append(address[key])
                if len(parts) == 2:
                    break

        return ", ".join(parts) if parts else "Unknown region"

    except Exception:
        return "Unknown region"
        
def load_image(image_path: str) -> tuple[np.ndarray, dict, str, int]:
    """
    Load a satellite image from disk.
    Returns: (array, meta, format, n_bands)
    """
    ext = os.path.splitext(image_path)[-1].lower()

    if ext in ['.tif', '.tiff']:
        if not RASTERIO_AVAILABLE:
            raise ImportError("rasterio is required for GeoTIFF files.")
        with rasterio.open(image_path) as src:
            array = src.read().astype(np.float32)
            meta = {
                "resolution": src.res,
                "crs":        str(src.crs),
                "width":      src.width,
                "height":     src.height,
                "dtype":      str(src.dtypes[0]),
                "n_bands":    src.count,
                "transform":  src.transform,
            }
        region = get_region_name(meta)
        meta["region_name"] = region
        print(f"  Region detected: {region}")
        return array, meta, "geotiff", meta["n_bands"]

    elif ext in ['.png', '.jpg', '.jpeg']:
        img = Image.open(image_path).convert('RGB')
        array = np.array(img, dtype=np.float32)     # (H, W, 3)
        array = np.transpose(array, (2, 0, 1))      # → (3, H, W)
        meta = {
            "resolution": "unknown",
            "crs": "unknown",
            "width": img.width,
            "height": img.height,
            "dtype": "uint8",
            "n_bands": 3,
        }
        return array, meta, "rgb", 3

    else:
        raise ValueError(f"Unsupported image format: {ext}")


def load_csv(csv_path: str) -> tuple[pd.DataFrame, dict]:
    """
    Load and summarize an optional CSV file.
    Returns: (dataframe, summary_dict)
    """
    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("CSV file is empty.")

    # Drop fully null columns silently
    df = df.dropna(axis=1, how='all')

    summary = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            summary[col] = {
                "mean": round(df[col].mean(), 4),
                "min": round(df[col].min(), 4),
                "max": round(df[col].max(), 4),
                "missing": int(df[col].isna().sum()),
            }
        else:
            summary[col] = {
                "unique_values": df[col].nunique(),
                "sample": df[col].dropna().iloc[0] if not df[col].dropna().empty else None,
            }

    return df, summary


def validate_image(array: np.ndarray, n_bands: int):
    """
    Check image is usable. Warns if NIR band is missing (no NDVI possible).
    """
    if array is None or array.size == 0:
        raise ValueError("Image array is empty.")

    if n_bands == 3:
        print("  [INFO] RGB image detected. NDVI/NDWI will be skipped (no NIR band).")
    elif n_bands >= 4:
        print(f"  [INFO] Multispectral image detected ({n_bands} bands). Full index computation available.")
    else:
        print(f"  [WARNING] Unusual band count: {n_bands}. Proceeding with caution.")

def detect_region_context(meta: dict) -> str:
    """
    Use CRS and resolution to infer broad geographic context.
    This gets passed to the LLM for better interpretation.
    """
    crs = str(meta.get("crs", "")).upper()
    res = meta.get("resolution", None)

    context_notes = []

    if "4326" in crs:
        context_notes.append("Image in geographic coordinates (WGS84)")
    elif "32" in crs:
        context_notes.append("Image in UTM projection")

    if res:
        if isinstance(res, tuple):
            pixel_size = res[0]
        else:
            pixel_size = res
        if pixel_size < 0.0002:
            context_notes.append("High resolution imagery (~10-20m/pixel)")
        elif pixel_size < 0.001:
            context_notes.append("Medium resolution imagery (~30m/pixel)")
        else:
            context_notes.append("Low resolution imagery (>100m/pixel)")

    return " | ".join(context_notes) if context_notes else "Unknown projection"

def build_input_context(
    image_path: str,
    csv_path: str = None,
    question: str = None,
) -> InputContext:
    """
    Main entry point. Takes file paths, returns a populated InputContext.
    """
    print("=== Input Handler ===")

    # --- Image (required) ---
    print(f"Loading image: {image_path}")
    try:
        array, meta, fmt, n_bands = load_image(image_path)
        validate_image(array, n_bands)
        print(f"  Shape: {array.shape} | Format: {fmt} | Bands: {n_bands}")
    except Exception as e:
        raise RuntimeError(f"Image loading failed: {e}")

    # --- CSV (optional) ---
    csv_df, csv_summary = None, None
    if csv_path:
        print(f"Loading CSV: {csv_path}")
        try:
            csv_df, csv_summary = load_csv(csv_path)
            print(f"  Rows: {len(csv_df)} | Columns: {list(csv_df.columns)}")
        except Exception as e:
            print(f"  [WARNING] CSV loading failed: {e}. Continuing without it.")

    # --- Question (optional) ---
    default_question = "Provide a full scientific interpretation of this image and data."
    user_question = question.strip() if question and question.strip() else default_question
    print(f"Question: {user_question}")

    # --- Build context ---
    context = InputContext(
    image_array=array,
    image_meta=meta,
    image_format=fmt,
    n_bands=n_bands,
    csv_df=csv_df,
    csv_summary=csv_summary,
    user_question=user_question,
)
    context.image_meta["region_context"] = detect_region_context(meta)

    print("=== InputContext ready ===\n")
    return context
