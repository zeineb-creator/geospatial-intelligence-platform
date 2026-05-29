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
    """
    Initialise Google Earth Engine safely.
    Returns True if successful, False otherwise.
    """

    try:
        import ee
    except ImportError:
        print("[GEE] earthengine-api not installed")
        return False

    # ── 1. Streamlit service account ─────────────────────────
    try:
        import streamlit as st

        sa_info = st.secrets.get("GEE_SERVICE_ACCOUNT", None)

        if sa_info:
            # if stored as string → convert
            if isinstance(sa_info, str):
                sa_info = json.loads(sa_info)

            credentials = ee.ServiceAccountCredentials(
                sa_info["streamlit-gee@capstone26-497821.iam.gserviceaccount.com"],
                sa_info["-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC5XBIx/PIcxALH\nvSyYEzkKW8AYTV1pxr06FLlI/+n6NbBnrvgnJbCVRUDT+eupmmJGxSw5acNF8Ig2\nKMaaejreH8hxRilqfaLYllWYl/f+KLAe23wzVksNCKuIbLfhqqG/Bi0EUn09Sc6R\nIt7MbqX0S718LuL6DSOkmSD+M4XjdVTGvtU/0F15ePVHWzw9A5TVKY1hgAZfxFoe\ny0aEKfs+hylneF//lIbI9Y3oG7Z8Z1yhd3u4MjhnbREijrM3hJd2jrWXDm+EYCC1\ntpXPD1cP8Ln0AcPHFVNcosXSljBiMM1eQzLW85bo6BFHPw6fr+Ry/0leKqUt65zF\ndPTVu6OlAgMBAAECggEADcPzqfix9kbsOe8nl6edzdIggu2D6W4T8XNGi1BODQHd\ntuV8klOvOHarKNLfmHnZoI3WCfF7yf7h7ru6fBBlvMc5eIgQXVM1O8Z2Vt26Ugqt\nga0lHR7kdfRV4MzKHx0v4+LOhwqlOF750zd2ulHhSzIwwT8uJgnBwuCIXbhv0gNt\nGhFxJmaRffoUVlFQ3CZsNsd+24/DDw9uqksEywQ8OZlmNTli8PnECEC4RThc1ByS\nePuqD/6YjOfZVjCuL8WhmJbAXrhuVCZlY5ude07SxWUx6JtI+9cDO3QK2zBN6eql\n7KHm/c1i3HmL8hiJeP1Q5h2TKXdROAYt5sdFUE9cYQKBgQDzO2JmFajtWLdDL1M8\n749Y/TPRebU62v/KggRMMXwhSgoBqWYEChS69t+rEzgPqsPXPEKH6Sn4UYAie763\n46xFHPn/iLfrubMgPe2RzOWPetfFIbwjuwtUdKPTO+e0kuyJaiUGoWsFf0qkIufl\n7ss4zB97WGzduVa1PaqRMHwo7QKBgQDDFvuxZS9i2CPGzAsyTJj7PcEnakIOpPw5\nrgvgz37Qp1iA+GztHRG0rRGZFzxvuQBT7EYz1v4fZSvhcnmzCFJ9eDpZFqi5hHht\nS5oKOB0m0/SKr88xh5g4UeqF2qGXaZ2WTx+c1uNEQcakVlY8H/5luZMZw1YEgUoq\nOnX2eMQmmQKBgQCSb44uJ5wsSN0MddGHPjLvQIGR+9RAOtkE8oHj3Wb+I7UiivoA\nNJGGflrqhAecZxLA4marrJS4C1k5aYbI7ykn8uoQDh+sq4BMSPxeax1J5ItDA6xh\nVprbnd2Dru0wqcP+dwkTlNr51Ej7yIgUxk9TQpExkWr5kjvBof6uqqIVgQKBgBwt\no6kZzmBei7xZGHzpZ2dSoiWJSYVH+05xfzG3hr+ojDYEq+cLvdT08ofEPWx9sjhs\n9i7irM3oFRB5Mm7TkuagiPz5MuGo6qQOuW8kb8F99+JYRnsG9MuduVwgGhr+nE5r\nm0tSFn1zaret6MLCdEJJLaAZAMGh0w6KZyN0ihBxAoGAeCVG0mtzpn6NCmamZEoK\n5guUORiOCtqySsHylQUsfn6pGWoqosNNIfstbOC5IhjQUFuQlJgOMWGGYWa2goNA\nFdzMa9gWHaf9MIU0NjppBcRa633CZpHyF5OEhvBKGlzWg2dST5TtmQRfeaZHgfhB\nIoP5MoTNi2/jKWB6Y5q4Q+s=\n-----END PRIVATE KEY-----\n"]   # ✅ FIX: correct field
            )

            ee.Initialize(credentials)

            print("[GEE] Initialized via Streamlit service account")
            return True

    except Exception as e:
        print(f"[GEE] Streamlit init failed: {e}")

    # ── 2. Local / default credentials ───────────────────────
    try:
        ee.Initialize()
        print("[GEE] Initialized via default credentials")
        return True

    except Exception as e:
        print(f"[GEE] Default credentials failed: {e}")
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
