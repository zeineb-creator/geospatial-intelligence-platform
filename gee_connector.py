"""
gee_connector.py — Optimized Google Earth Engine connector
===========================================================
Fast preview-based streaming (thumb tiles) + geemap integration
"""

import ee
import numpy as np
import time
import json

# Optional but recommended
import geemap.foliumap as geemap


# ─────────────────────────────────────────────
# INIT GEE
# ─────────────────────────────────────────────

def init_gee():
    try:
        ee.Initialize()
        print("[GEE] Initialized")
        return True
    except Exception as e:
        print("[GEE] Init failed:", e)
        return False


# ─────────────────────────────────────────────
# SENSOR CONFIGS (FIXED SCALING)
# ─────────────────────────────────────────────

SENSOR_CONFIGS = {
    "Sentinel-2 SR": {
        "collection": "COPERNICUS/S2_SR_HARMONIZED",
        "bands": ["B4", "B3", "B2", "B8"],  # RGB + NIR
        "scale": 0.0001,
        "cloud": "CLOUDY_PIXEL_PERCENTAGE",
        "start": 2017,
    },

    "Landsat 8/9": {
        "collection": "LANDSAT/LC08/C02/T1_L2",
        "bands": ["SR_B4", "SR_B3", "SR_B2", "SR_B5"],
        "scale": 0.0000275,
        "offset": -0.2,
        "cloud": "CLOUD_COVER",
        "start": 2013,
    },

    "Landsat 7": {
        "collection": "LANDSAT/LE07/C02/T1_L2",
        "bands": ["SR_B3", "SR_B2", "SR_B1", "SR_B4"],
        "scale": 0.0000275,
        "offset": -0.2,
        "cloud": "CLOUD_COVER",
        "start": 1999,
    }
}


def auto_sensor(year):
    if year >= 2017:
        return "Sentinel-2 SR"
    elif year >= 2013:
        return "Landsat 8/9"
    return "Landsat 7"


# ─────────────────────────────────────────────
# AOI
# ─────────────────────────────────────────────

def aoi(lon, lat, km):
    return ee.Geometry.Point([lon, lat]).buffer(km * 1000).bounds()


# ─────────────────────────────────────────────
# COLLECTION BUILDER (IMPROVED)
# ─────────────────────────────────────────────

def build_collection(cfg, aoi_geom, year):
    start = f"{year}-01-01"
    end = f"{year+1}-01-01"

    col = (
        ee.ImageCollection(cfg["collection"])
        .filterBounds(aoi_geom)
        .filterDate(start, end)
        .sort("CLOUDY_PIXEL_PERCENTAGE")   # ✅ requested improvement
    )

    # fallback cloud handling
    col = col.filter(ee.Filter.lt(cfg["cloud"], 80))

    return col.select(cfg["bands"])


# ─────────────────────────────────────────────
# SCALING (FIXED LANDSAT)
# ─────────────────────────────────────────────

def apply_scaling(img, cfg):
    img = img.multiply(cfg["scale"])

    # Landsat offset fix
    if "offset" in cfg:
        img = img.add(cfg["offset"])

    return img


# ─────────────────────────────────────────────
# FAST THUMB STREAM (REPLACES DOWNLOAD)
# ─────────────────────────────────────────────

def get_thumb(image, region, scale=30):
    """
    Much faster than getDownloadURL.
    Returns a PNG preview URL (streamable).
    """

    params = {
        "region": region,
        "dimensions": 1024,
        "format": "png",
        "min": 0,
        "max": 0.3,
        "bands": None
    }

    return image.getThumbURL(params)


# ─────────────────────────────────────────────
# MAIN FETCH (OPTIMIZED PIPELINE)
# ─────────────────────────────────────────────

def fetch_image_preview(lat, lon, year, buffer_km=5, sensor=None):
    sensor = sensor or auto_sensor(year)
    cfg = SENSOR_CONFIGS[sensor]

    print(f"[GEE] Sensor={sensor}, Year={year}")

    region = aoi(lon, lat, buffer_km)

    # collection
    col = build_collection(cfg, region, year)

    count = col.size().getInfo()
    print("[GEE] Scenes:", count)

    if count == 0:
        print("[GEE] No data")
        return None

    # median composite + requested fix
    image = col.sort("CLOUDY_PIXEL_PERCENTAGE").first()  # ✅ REQUIRED FIX

    image = apply_scaling(image, cfg)

    # clip
    image = image.clip(region)

    # FAST STREAMING PREVIEW
    thumb_url = get_thumb(image, region.getInfo())

    meta = {
        "sensor": sensor,
        "year": year,
        "scenes": count,
        "mode": "thumb_stream"
    }

    print("[GEE] Thumb URL ready")
    return thumb_url, meta


# ─────────────────────────────────────────────
# GEEMAP VISUALIZATION (OPTIONAL UI)
# ─────────────────────────────────────────────

def create_map(lat, lon, year):
    Map = geemap.Map(center=[lat, lon], zoom=10)

    sensor = auto_sensor(year)
    cfg = SENSOR_CONFIGS[sensor]

    region = aoi(lon, lat, 5)
    col = build_collection(cfg, region, year)

    image = col.sort("CLOUDY_PIXEL_PERCENTAGE").first()
    image = apply_scaling(image, cfg).clip(region)

    vis = {
        "bands": cfg["bands"][:3],
        "min": 0,
        "max": 0.3
    }

    Map.addLayer(image, vis, "Composite")
    Map.addLayer(region, {}, "AOI")

    return Map
