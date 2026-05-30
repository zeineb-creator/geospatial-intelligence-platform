"""
gee_connector.py — Google Earth Engine direct image fetcher
============================================================
"""

import os
import json
import tempfile
import numpy as np


# ─────────────────────────────────────────────────────────────
# GEE INITIALISATION
# ─────────────────────────────────────────────────────────────

def init_gee() -> bool:
    try:
        import ee
    except ImportError:
        print("[GEE] earthengine-api not installed")
        return False

    try:
        import streamlit as st
        sa_info = st.secrets.get("GEE_SERVICE_ACCOUNT", None)
        if not sa_info:
            print("[GEE] No GEE_SERVICE_ACCOUNT in secrets")
            return False

        sa_dict = {
            "type":                        sa_info.get("type", "service_account"),
            "project_id":                  sa_info["project_id"],
            "private_key_id":              sa_info["private_key_id"],
            "private_key":                 sa_info["private_key"].replace("\\n", "\n"),
            "client_email":                sa_info["client_email"],
            "client_id":                   sa_info["client_id"],
            "auth_uri":                    sa_info.get("auth_uri", "https://accounts.google.com/o/oauth2/auth"),
            "token_uri":                   sa_info.get("token_uri", "https://oauth2.googleapis.com/token"),
            "auth_provider_x509_cert_url": sa_info.get("auth_provider_x509_cert_url", "https://www.googleapis.com/oauth2/v1/certs"),
            "client_x509_cert_url":        sa_info.get("client_x509_cert_url", ""),
        }

        credentials = ee.ServiceAccountCredentials(
            email=sa_dict["client_email"],
            key_data=json.dumps(sa_dict),
        )
        ee.Initialize(credentials, project=sa_dict["project_id"])
        print("[GEE] Initialized via Streamlit service account")
        return True

    except Exception as e:
        print(f"[GEE] Streamlit auth failed: {e}")

    try:
        import ee
        ee.Initialize()
        print("[GEE] Initialized via default credentials")
        return True
    except Exception as e:
        print(f"[GEE] Default auth failed: {e}")

    return False


def gee_available() -> bool:
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
        "collection":   "COPERNICUS/S2_SR_HARMONIZED",
        "bands":        ["B2", "B3", "B4", "B8", "B11", "B12"],
        "scale_factor": 0.0001,
        "resolution":   10,
        "cloud_prop":   "CLOUDY_PIXEL_PERCENTAGE",
        "cloud_max":    30,
    },
    "Landsat 8/9": {
        "collection":   "LANDSAT/LC08/C02/T1_L2",
        "bands":        ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
        "scale_factor": 0.0000275,
        "offset":       -0.2,
        "resolution":   30,
        "cloud_prop":   "CLOUD_COVER",
        "cloud_max":    30,
    },
    "Landsat 5 TM": {
        "collection":   "LANDSAT/LT05/C02/T1_L2",
        "bands":        ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
        "scale_factor": 0.0000275,
        "offset":       -0.2,
        "resolution":   30,
        "cloud_prop":   "CLOUD_COVER",
        "cloud_max":    30,
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
# IMAGE FETCHING
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
    cfg    = SENSOR_CONFIGS[sensor]

    print(f"[GEE] Fetching {sensor} {year}")

    point = ee.Geometry.Point([lon, lat])
    aoi   = point.buffer(buffer_km * 1000).bounds()

    start = f"{year}-{month_start:02d}-01"
    end   = f"{year}-{month_end:02d}-28"

    # Try progressively relaxed cloud filters
    col = None
    for cloud_max in [cfg["cloud_max"], 50, 80]:
        col = (
            ee.ImageCollection(cfg["collection"])
            .filterBounds(aoi)
            .filterDate(start, end)
            .filter(ee.Filter.lt(cfg["cloud_prop"], cloud_max))
            .select(cfg["bands"])
        )
        count = col.size().getInfo()
        print(f"[GEE] Scenes found (cloud<{cloud_max}%): {count}")
        if count > 0:
            break

    if col is None or col.size().getInfo() == 0:
        print("[GEE] No scenes found")
        return None

    # Build median composite — do NOT apply scaling here,
    # scaling is applied AFTER getDownloadURL on the raw values
    image = col.median()
    image = image.clip(aoi)

    try:
        import requests, zipfile, io, rasterio, time

        # Use 60m to stay well under GEE's download size limit
        download_scale = 60

        params = {
            "scale":  download_scale,
            "region": aoi.getInfo(),
            "format": "GEO_TIFF",
            "bands":  cfg["bands"],
            "crs":    "EPSG:4326",
        }

        print(f"[GEE] Requesting download at {download_scale}m...")
        url = image.getDownloadURL(params)
        print(f"[GEE] URL obtained, downloading...")

        # Retry logic
        r = None
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=300)
                if r.status_code == 200:
                    break
                print(f"[GEE] HTTP {r.status_code} on attempt {attempt+1}, retrying...")
                time.sleep(3)
            except requests.exceptions.Timeout:
                print(f"[GEE] Timeout on attempt {attempt+1}")
                time.sleep(3)

        if r is None or r.status_code != 200:
            print(f"[GEE] Download failed after retries")
            return None

        print(f"[GEE] Downloaded {len(r.content)/1024:.1f} KB")

        # Validate it's actually a zip (starts with PK magic bytes)
        if len(r.content) < 4 or r.content[:2] != b'PK':
            print(f"[GEE] Not a zip file. Content preview: {r.content[:300]}")
            return None

        # GEE zip contains one TIF per band, named by band
        bands_data = []
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            tif_names = sorted([n for n in z.namelist() if n.endswith(".tif")])
            print(f"[GEE] ZIP contains {len(tif_names)} files: {tif_names}")

            for name in tif_names:
                # Must write to real temp file — rasterio needs seekable file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
                    tmp.write(z.read(name))
                    tmp_path = tmp.name
                try:
                    with rasterio.open(tmp_path) as src:
                        band_data = src.read(1).astype(np.float32)
                        bands_data.append(band_data)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

        if not bands_data:
            print("[GEE] No bands extracted")
            return None

        # Stack bands into (C, H, W)
        array = np.stack(bands_data, axis=0)

        # Apply scaling NOW (on raw DN values from the download)
        array = array * cfg["scale_factor"]
        if "offset" in cfg:
            array = array + cfg["offset"]

        array = np.clip(array, 0.0, 1.0)

        meta = {
            "sensor": sensor,
            "year":   year,
            "source": "GEE",
        }

        print(f"[GEE] Success — shape: {array.shape}, "
              f"range: [{array.min():.3f}, {array.max():.3f}]")
        return array, meta

    except zipfile.BadZipFile:
        print("[GEE] BadZipFile — GEE returned an error page instead of a zip")
        if r is not None:
            print(f"[GEE] Response preview: {r.content[:500]}")
        return None

    except Exception as e:
        print(f"[GEE] Download error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────
# SAVE FUNCTION
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


def fetch_and_save(lat, lon, year, buffer_km=5.0, sensor=None, **kwargs):
    """Fetch image and save to a temporary GeoTIFF file."""
    result = fetch_image_as_array(
        lat=lat, lon=lon, year=year,
        buffer_km=buffer_km, sensor=sensor,
    )
    if result is None:
        return None

    array, meta = result

    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        path = tmp.name

    return save_array_as_geotiff(array, meta, path)
