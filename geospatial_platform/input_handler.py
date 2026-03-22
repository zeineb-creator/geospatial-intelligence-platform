
import numpy as np
import pandas as pd
from PIL import Image
from dataclasses import dataclass
import os
import sys
sys.path.append("/kaggle/working")

from geospatial_platform.context import InputContext

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


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
            array = src.read().astype(np.float32)   # (bands, H, W)
            meta = {
                "resolution": src.res,
                "crs": str(src.crs),
                "width": src.width,
                "height": src.height,
                "dtype": str(src.dtypes[0]),
                "n_bands": src.count,
            }
        return array, meta, "geotiff", src.count

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

    print("=== InputContext ready ===\n")
    return context
