"""
gee_connector.py — Google Earth Engine direct image fetcher
============================================================
Fetches satellite imagery directly from GEE using a service account,
saving the user from manual export/upload.

Supports:
  - Sentinel-2 L2A (10m bands, 6-band export)
  - Landsat 8/9 OLI (30m, 6-band export)
  - Landsat 5 TM (30m, 6-band export, for historical imagery)

Authentication:
  Streamlit Cloud: service account JSON stored in st.secrets["GEE_SERVICE_ACCOUNT"]
  Local/Kaggle:    GOOGLE_APPLICATION_CREDENTIALS env var, or ee.Authenticate()
"""

import os
import json
import tempfile
import numpy as np


# ── GEE initialisation ────────────────────────────────────────────────────────

def init_gee() -> bool:
    """
    Initialise GEE. Returns True if successful, False if unavailable.
    Tries Streamlit secrets first, then env var, then interactive auth.
    """
    try:
        import ee
    except ImportError:
        print("  [GEE] earthengine-api not installed — skipping GEE integration")
        return False

    # Try service account from Streamlit secrets
    try:
        import streamlit as st
        sa_info = st.secrets.get("GEE_SERVICE_ACCOUNT", None)
        if sa_info:
            if isinstance(sa_info, str):
                sa_info = json.loads(sa_info)
            credentials = ee.ServiceAccountCredentials(
                email=sa_info["client_email"],
                key_data=json.dumps(sa_info),
            )
            ee.Initialize(credentials)
            print("  [GEE] Initialised via Streamlit service account")
            return True
    except Exception as e:
        print(f"  [GEE] Streamlit secret init failed: {e}")

    # Try default credentials (local dev / Kaggle)
    try:
        import ee
        ee.Initialize()
        print("  [GEE] Initialised via default credentials")
        return True
    except Exception as e:
        print(f"  [GEE] Default credentials failed: {e}")
        return False


def gee_available() -> bool:
    """Quick check — does not re-initialise."""
    try:
        import ee
        ee.Number(1).getInfo()
        return True
    except Exception:
        return False


# ── Sensor configurations ─────────────────────────────────────────────────────

SENSOR_CONFIGS = {
    "Sentinel-2 L2A": {
        "collection":  "COPERNICUS/S2_SR_HARMONIZED",
        "bands":       ["B2", "B3", "B4", "B8", "B11", "B12"],
        "band_names":  ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"],
        "scale_factor": 0.0001,
        "resolution":  10,
        "min_year":    2017,
        "cloud_prop":  "CLOUDY_PIXEL_PERCENTAGE",
        "cloud_max":   20,
    },
    "Landsat 8/9": {
        "collection":  "LANDSAT/LC09/C02/T1_L2",  # use LC08 as fallback
        "bands":       ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
        "band_names":  ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"],
        "scale_factor": 0.0000275,
        "offset":      -0.2,
        "resolution":  30,
        "min_year":    2013,
        "cloud_prop":  "CLOUD_COVER",
        "cloud_max":   20,
    },
    "Landsat 5 TM": {
        "collection":  "LANDSAT/LT05/C02/T1_L2",
        "bands":       ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
        "band_names":  ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"],
        "scale_factor": 0.0000275,
        "offset":      -0.2,
        "resolution":  30,
        "min_year":    1984,
        "max_year":    2013,
        "cloud_prop":  "CLOUD_COVER",
        "cloud_max":   20,
    },
}


def auto_select_sensor(year: int) -> str:
    """Choose best sensor for a given year."""
    if year >= 2017:
        return "Sentinel-2 L2A"
    elif year >= 2013:
        return "Landsat 8/9"
    else:
        return "Landsat 5 TM"


# ── Image fetching ────────────────────────────────────────────────────────────

def fetch_image_as_array(
    lat: float,
    lon: float,
    year: int,
    month_start: int = 1,
    month_end: int = 12,
    sensor: str = None,
    buffer_km: float = 5.0,
) -> tuple[np.ndarray, dict] | None:
    """
    Fetch a cloud-free composite from GEE for a given location and year.
    Returns (array, meta) where array is (6, H, W) float32 in reflectance [0,1].
    Returns None if GEE unavailable or no valid scenes found.

    Parameters:
        lat, lon      : Centre coordinates
        year          : Year of interest
        month_start   : Start month (1-12)
        month_end     : End month (1-12)
        sensor        : Sensor name (auto-selected if None)
        buffer_km     : Area radius in km (~5km gives ~333x333 px at 30m)
    """
    try:
        import ee
    except ImportError:
        return None

    sensor = sensor or auto_select_sensor(year)
    cfg    = SENSOR_CONFIGS[sensor]

    print(f"  [GEE] Fetching {sensor} | {year}-{month_start:02d} to {year}-{month_end:02d}")
    print(f"  [GEE] Location: lat={lat:.4f}, lon={lon:.4f}, buffer={buffer_km}km")

    # Define AOI
    point  = ee.Geometry.Point([lon, lat])
    aoi    = point.buffer(buffer_km * 1000).bounds()

    # Date range
    start  = f"{year}-{month_start:02d}-01"
    end_m  = month_end + 1 if month_end < 12 else 1
    end_y  = year if month_end < 12 else year + 1
    end    = f"{end_y}-{end_m:02d}-01"

    # Build collection
    col = (ee.ImageCollection(cfg["collection"])
           .filterBounds(aoi)
           .filterDate(start, end)
           .filter(ee.Filter.lt(cfg["cloud_prop"], cfg["cloud_max"]))
           .select(cfg["bands"]))

    count = col.size().getInfo()
    print(f"  [GEE] Found {count} valid scenes")
    if count == 0:
        # Relax cloud threshold and try again
        col = (ee.ImageCollection(cfg["collection"])
               .filterBounds(aoi)
               .filterDate(start, end)
               .select(cfg["bands"]))
        count = col.size().getInfo()
        if count == 0:
            print("  [GEE] No scenes found after relaxing cloud filter")
            return None

    # Median composite
    image = col.median().multiply(cfg["scale_factor"])
    if "offset" in cfg:
        image = image.add(cfg["offset"])

    # Clip to AOI
    image = image.clip(aoi)

    # Export to numpy via getDownloadURL (small AOI only)
    try:
        import requests, zipfile, io, rasterio

        url = image.getDownloadURL({
            "scale":   cfg["resolution"],
            "region":  aoi.getInfo(),
            "format":  "GEO_TIFF",
            "bands":   cfg["bands"],
        })

        print(f"  [GEE] Downloading image...")
        r = requests.get(url, timeout=120)
        r.raise_for_status()

        # GEE returns a zip with one TIF per band
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            tif_names = [n for n in z.namelist() if n.endswith('.tif')]
            bands = []
            crs, transform, width, height = None, None, None, None
            for tif_name in sorted(tif_names):
                with z.open(tif_name) as tif_file:
                    with rasterio.open(tif_file) as src:
                        bands.append(src.read(1).astype(np.float32))
                        if crs is None:
                            crs       = str(src.crs)
                            transform = src.transform
                            width     = src.width
                            height    = src.height

        array = np.stack(bands, axis=0)  # (6, H, W)
        array = np.clip(array, 0, 1)

        meta = {
            "crs":        crs,
            "transform":  transform,
            "width":      width,
            "height":     height,
            "resolution": (cfg["resolution"] / 111320, cfg["resolution"] / 111320),
            "n_bands":    len(bands),
            "sensor":     sensor,
            "year":       year,
            "source":     "Google Earth Engine",
        }

        print(f"  [GEE] Downloaded: {array.shape}, range [{array.min():.3f}, {array.max():.3f}]")
        return array, meta

    except Exception as e:
        print(f"  [GEE] Download failed: {e}")
        return None


def save_array_as_geotiff(array: np.ndarray, meta: dict, output_path: str) -> str:
    """Save a numpy array as a GeoTIFF for downstream processing."""
    try:
        import rasterio
        from rasterio.transform import from_bounds

        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=array.shape[1],
            width=array.shape[2],
            count=array.shape[0],
            dtype=np.float32,
            crs=meta.get("crs", "EPSG:4326"),
            transform=meta.get("transform"),
        ) as dst:
            dst.write(array)
        return output_path
    except Exception as e:
        print(f"  [GEE] Save failed: {e}")
        return None


def fetch_and_save(
    lat: float, lon: float, year: int,
    month_start: int = 1, month_end: int = 12,
    sensor: str = None, buffer_km: float = 5.0,
) -> str | None:
    """
    Fetch image from GEE and save to a temp GeoTIFF.
    Returns temp file path or None on failure.
    """
    result = fetch_image_as_array(lat, lon, year, month_start, month_end, sensor, buffer_km)
    if result is None:
        return None
    array, meta = result

    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        path = tmp.name
    return save_array_as_geotiff(array, meta, path)
