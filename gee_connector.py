"""
gee_connector.py — Google Earth Engine direct image fetcher
============================================================
"""

import os
import json
import tempfile
import numpy as np


# ─────────────────────────────────────────────────────────────
# GEE INITIALISATION (FIXED)
# ─────────────────────────────────────────────────────────────

def init_gee() -> bool:
    try:
        import ee
    except ImportError:
        print("[GEE] earthengine-api not installed")
        return False

    # 1) Streamlit secrets (recommended)
    try:
        import streamlit as st
        sa_info = st.secrets.get("GEE_SERVICE_ACCOUNT", None)

        if sa_info:
            import json
            credentials = ee.ServiceAccountCredentials(
                email=sa_info["client_email"],        # ✅ fix
                key_data=json.dumps(dict(sa_info)),   # ✅ fix
            )
            ee.Initialize(credentials)
            print("[GEE] Initialized via Streamlit service account")
            return True

    except Exception as e:
        print(f"[GEE] Streamlit auth failed: {e}")

    # 2) Environment fallback
    try:
        ee.Initialize()
        print("[GEE] Initialized via default credentials")
        return True
    except Exception as e:
        print(f"[GEE] Default auth failed: {e}")

    return False

def gee_available() -> bool:
    """Quick check without reinitializing."""
    try:
        import ee
        ee.Number(1).getInfo()
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────
# SENSOR CONFIGS
# ─────────────────────────────────────────────────────────────

SENSOR_CONFIGS = {
    "Sentinel-2 L2A": {
        "collection": "COPERNICUS/S2_SR_HARMONIZED",
        "bands": ["B2", "B3", "B4", "B8", "B11", "B12"],
        "scale_factor": 0.0001,
        "resolution": 10,
        "cloud_prop": "CLOUDY_PIXEL_PERCENTAGE",
        "cloud_max": 20,
    },
    "Landsat 8/9": {
        "collection": "LANDSAT/LC09/C02/T1_L2",
        "bands": ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
        "scale_factor": 0.0000275,
        "offset": -0.2,
        "resolution": 30,
        "cloud_prop": "CLOUD_COVER",
        "cloud_max": 20,
    },
    "Landsat 5 TM": {
        "collection": "LANDSAT/LT05/C02/T1_L2",
        "bands": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
        "scale_factor": 0.0000275,
        "offset": -0.2,
        "resolution": 30,
        "cloud_prop": "CLOUD_COVER",
        "cloud_max": 20,
    },
}


def auto_select_sensor(year: int) -> str:
    if year >= 2017:
        return "Sentinel-2 L2A"
    elif year >= 2013:
        return "Landsat 8/9"
    else:
        return "Landsat 5 TM"


# ─────────────────────────────────────────────────────────────
# IMAGE FETCHING (UNCHANGED CORE LOGIC, SAFE)
# ─────────────────────────────────────────────────────────────

def fetch_image_as_array(
    lat: float,
    lon: float,
    year: int,
    month_start: int = 1,
    month_end: int = 12,
    sensor: str = None,
    buffer_km: float = 5.0,
):
    try:
        import ee
    except ImportError:
        return None

    sensor = sensor or auto_select_sensor(year)
    cfg = SENSOR_CONFIGS[sensor]

    print(f"[GEE] Fetching {sensor} {year}")

    point = ee.Geometry.Point([lon, lat])
    aoi = point.buffer(buffer_km * 1000).bounds()

    start = f"{year}-{month_start:02d}-01"
    end = f"{year}-{month_end:02d}-28"

    col = (
        ee.ImageCollection(cfg["collection"])
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt(cfg["cloud_prop"], cfg["cloud_max"]))
        .select(cfg["bands"])
    )

    if col.size().getInfo() == 0:
        print("[GEE] No scenes found")
        return None

    image = col.median().multiply(cfg["scale_factor"])
    if "offset" in cfg:
        image = image.add(cfg["offset"])

    image = image.clip(aoi)

    try:
        import requests, zipfile, io, rasterio

        url = image.getDownloadURL({
            "scale": cfg["resolution"],
            "region": aoi.getInfo(),
            "format": "GEO_TIFF",
            "bands": cfg["bands"],
        })

        r = requests.get(url, timeout=120)
        r.raise_for_status()

        bands = []
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            for name in sorted(z.namelist()):
                if name.endswith(".tif"):
                    with z.open(name) as f:
                        with rasterio.open(f) as src:
                            bands.append(src.read(1).astype(np.float32))

        array = np.stack(bands, axis=0)
        array = np.clip(array, 0, 1)

        meta = {
            "sensor": sensor,
            "year": year,
            "source": "GEE",
        }

        print(f"[GEE] Success {array.shape}")
        return array, meta

    except Exception as e:
        print(f"[GEE] Download error: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# SAVE FUNCTION (UNCHANGED BUT SAFE)
# ─────────────────────────────────────────────────────────────

def save_array_as_geotiff(array: np.ndarray, meta: dict, output_path: str):
    try:
        import rasterio

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=array.shape[1],
            width=array.shape[2],
            count=array.shape[0],
            dtype=np.float32,
        ) as dst:
            dst.write(array)

        return output_path

    except Exception as e:
        print(f"[GEE] Save failed: {e}")
        return None


def fetch_and_save(*args, **kwargs):
    result = fetch_image_as_array(*args, **kwargs)
    if result is None:
        return None

    array, meta = result

    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        path = tmp.name

    return save_array_as_geotiff(array, meta, path)
