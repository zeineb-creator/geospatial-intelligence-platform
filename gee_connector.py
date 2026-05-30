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
        "collection":  "COPERNICUS/S2_SR_HARMONIZED",
        "bands":       ["B2", "B3", "B4", "B8", "B11", "B12"],
        "scale_factor": 0.0001,
        "resolution":  10,
        "cloud_prop":  "CLOUDY_PIXEL_PERCENTAGE",
        "cloud_max":   30,
    },
    "Landsat 8/9": {
        "collection":  "LANDSAT/LC08/C02/T1_L2",
        "bands":       ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
        "scale_factor": 0.0000275,
        "offset":      -0.2,
        "resolution":  30,
        "cloud_prop":  "CLOUD_COVER",
        "cloud_max":   30,
    },
    "Landsat 5 TM": {
        "collection":  "LANDSAT/LT05/C02/T1_L2",
        "bands":       ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
        "scale_factor": 0.0000275,
        "offset":      -0.2,
        "resolution":  30,
        "cloud_prop":  "CLOUD_COVER",
        "cloud_max":   30,
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

    col = (
        ee.ImageCollection(cfg["collection"])
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt(cfg["cloud_prop"], cfg["cloud_max"]))
        .select(cfg["bands"])
    )

    count = col.size().getInfo()
    print(f"[GEE] Scenes found: {count}")
    if count == 0:
        print("[GEE] No scenes found")
        return None

    # Build composite and apply scaling
    image = col.median().multiply(cfg["scale_factor"])
    if "offset" in cfg:
        image = image.add(cfg["offset"])
    image = image.clip(aoi)

    # ── Download via getDownloadURL with increased scale to stay under size limit
    try:
        import requests, zipfile, io, rasterio, os

        # Use coarser resolution to avoid the zip size limit
        # Sentinel-2: use 60m instead of 10m; Landsat: use 60m instead of 30m
        download_scale = max(cfg["resolution"], 60)

        params = {
            "scale":       download_scale,
            "region":      aoi.getInfo(),
            "format":      "GEO_TIFF",
            "bands":       cfg["bands"],
            "crs":         "EPSG:4326",
        }

        print(f"[GEE] Requesting download at {download_scale}m resolution...")
        url = image.getDownloadURL(params)
        print(f"[GEE] Download URL obtained, fetching...")

        r = requests.get(url, timeout=300)

        # Check if response is actually a zip
        if r.status_code != 200:
            print(f"[GEE] HTTP error: {r.status_code} — {r.text[:500]}")
            return None

        content_type = r.headers.get("Content-Type", "")
        print(f"[GEE] Response: {r.status_code}, size={len(r.content)/1024:.1f}KB, type={content_type}")

        if "zip" not in content_type and not r.content[:2] == b'PK':
            print(f"[GEE] Not a zip file. Response preview: {r.content[:500]}")
            return None

        bands = []
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            tif_names = sorted([n for n in z.namelist() if n.endswith(".tif")])
            print(f"[GEE] ZIP contains: {tif_names}")

            for name in tif_names:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
                    tmp.write(z.read(name))
                    tmp_path = tmp.name
                try:
                    with rasterio.open(tmp_path) as src:
                        bands.append(src.read(1).astype(np.float32))
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

        if not bands:
            print("[GEE] No bands extracted from ZIP")
            return None

        array = np.stack(bands, axis=0)
        array = np.clip(array, 0, 1)

        meta = {"sensor": sensor, "year": year, "source": "GEE"}
        print(f"[GEE] Success — array shape: {array.shape}")
        return array, meta

    except zipfile.BadZipFile:
        print(f"[GEE] BadZipFile — response was not a zip. Likely a GEE size/quota error.")
        print(f"[GEE] Response content preview: {r.content[:500]}")
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


def fetch_and_save(*args, **kwargs):
    result = fetch_image_as_array(*args, **kwargs)
    if result is None:
        return None

    array, meta = result

    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        path = tmp.name

    return save_array_as_geotiff(array, meta, path)
