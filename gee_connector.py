"""
gee_connector.py — Optimized Google Earth Engine connector
===========================================================
Fast preview-based streaming + actual data download
"""

import ee
import numpy as np
import tempfile
import io
import zipfile
import requests
import time
import os


# ─────────────────────────────────────────────
# INIT GEE (SIMPLIFIED)
# ─────────────────────────────────────────────

def init_gee():
    import ee

    # 1. Try default (rarely works on Streamlit)
    try:
        ee.Initialize()
        print("[GEE] Default init OK")
        return True
    except Exception as e:
        print("[GEE] Default init failed:", e)

    # 2. Streamlit secrets (MAIN METHOD)
    try:
        import streamlit as st
        from oauth2client.service_account import ServiceAccountCredentials

        sa = st.secrets["GEE_SERVICE_ACCOUNT"]

        credentials = ServiceAccountCredentials(
            sa["client_email"],
            key_data=sa["private_key"]
        )

        ee.Initialize(credentials)

        print("[GEE] Service account init OK")
        return True

    except Exception as e:
        print("[GEE] Service account init failed:", e)

    print("[GEE] ❌ INIT FAILED")
    return False


def gee_available():
    """Check if GEE is available."""
    try:
        import ee
        ee.Number(1).getInfo()
        return True
    except:
        return False


# ─────────────────────────────────────────────
# SENSOR CONFIGS
# ─────────────────────────────────────────────

SENSOR_CONFIGS = {
    "Sentinel-2 SR": {
        "collection": "COPERNICUS/S2_SR_HARMONIZED",
        "bands": ["B4", "B3", "B2", "B8"],  # RGB + NIR
        "scale": 0.0001,
        "resolution": 10,
        "cloud": "CLOUDY_PIXEL_PERCENTAGE",
        "start": 2017,
    },
    "Landsat 8/9": {
        "collection": "LANDSAT/LC08/C02/T1_L2",
        "bands": ["SR_B4", "SR_B3", "SR_B2", "SR_B5"],
        "scale": 0.0000275,
        "offset": -0.2,
        "resolution": 30,
        "cloud": "CLOUD_COVER",
        "start": 2013,
    },
    "Landsat 7": {
        "collection": "LANDSAT/LE07/C02/T1_L2",
        "bands": ["SR_B3", "SR_B2", "SR_B1", "SR_B4"],
        "scale": 0.0000275,
        "offset": -0.2,
        "resolution": 30,
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
# COLLECTION BUILDER
# ─────────────────────────────────────────────

def build_collection(cfg, aoi_geom, year, cloud_max=80):
    start = f"{year}-01-01"
    end = f"{year+1}-01-01"

    col = (
        ee.ImageCollection(cfg["collection"])
        .filterBounds(aoi_geom)
        .filterDate(start, end)
        .filter(ee.Filter.lt(cfg["cloud"], cloud_max))
    )
    
    return col.select(cfg["bands"])


# ─────────────────────────────────────────────
# SCALING
# ─────────────────────────────────────────────

def apply_scaling(img, cfg):
    img = img.multiply(cfg["scale"])
    if "offset" in cfg:
        img = img.add(cfg["offset"])
    return img


# ─────────────────────────────────────────────
# GET BEST IMAGE (LOWEST CLOUD COVER)
# ─────────────────────────────────────────────

def get_best_image(lat, lon, year, buffer_km=5, sensor=None):
    """Get the least cloudy image for the location/year."""
    sensor = sensor or auto_sensor(year)
    cfg = SENSOR_CONFIGS[sensor]
    
    region = aoi(lon, lat, buffer_km)
    
    # Try with relaxed cloud filters
    for cloud_max in [30, 60, 80]:
        try:
            col = build_collection(cfg, region, year, cloud_max=cloud_max)
            count = col.size().getInfo()
            if count > 0:
                print(f"[GEE] Found {count} scenes with cloud < {cloud_max}%")
                # Get the least cloudy image
                image = col.sort(cfg["cloud"]).first()
                return image, cfg, region
        except Exception as e:
            print(f"[GEE] Error checking cloud level {cloud_max}: {e}")
    
    print(f"[GEE] No scenes found for {sensor} {year}")
    return None, None, None


# ─────────────────────────────────────────────
# FAST PREVIEW URL (FOR UI)
# ─────────────────────────────────────────────

def get_preview_url(image, region, vis_params=None):
    """Get a fast preview URL for the image."""
    if vis_params is None:
        vis_params = {
            "min": 0,
            "max": 0.3,
            "bands": ["B4", "B3", "B2"]  # RGB
        }
    
    try:
        # Get region as GeoJSON
        region_json = region.getInfo()
        
        params = {
            "region": region_json,
            "dimensions": 1024,
            "format": "png"
        }
        params.update(vis_params)
        
        return image.getThumbURL(params)
    except Exception as e:
        print(f"[GEE] Preview URL error: {e}")
        return None


# ─────────────────────────────────────────────
# DOWNLOAD ACTUAL DATA
# ─────────────────────────────────────────────

def download_image_as_array(image, region, cfg, scale=30):
    """Download the actual image data as a numpy array."""
    try:
        # Get region as GeoJSON
        region_json = region.getInfo()
        
        # Build download parameters
        params = {
            "scale": scale,
            "region": region_json,
            "format": "GEO_TIFF",
            "crs": "EPSG:4326",
        }
        
        print(f"[GEE] Downloading at {scale}m resolution...")
        url = image.getDownloadURL(params)
        
        # Download with retry
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=300)
                
                if response.status_code != 200:
                    print(f"[GEE] Download failed: HTTP {response.status_code}")
                    if attempt < 2:
                        time.sleep(2)
                        continue
                    return None
                
                # Check if it's a zip file
                if len(response.content) < 100 or response.content[:2] != b'PK':
                    print("[GEE] Response is not a valid zip file")
                    return None
                
                # Process zip file
                import rasterio
                from rasterio.io import MemoryFile
                
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    tif_files = [f for f in z.namelist() if f.endswith('.tif')]
                    if not tif_files:
                        print("[GEE] No TIF files in zip")
                        return None
                    
                    # Read the first TIF (should contain all bands)
                    with z.open(tif_files[0]) as tif_file:
                        with MemoryFile(tif_file) as memfile:
                            with memfile.open() as src:
                                array = src.read().astype(np.float32)
                    
                    # Apply scaling
                    array = array * cfg["scale"]
                    if "offset" in cfg:
                        array = array + cfg["offset"]
                    
                    array = np.clip(array, 0.0, 1.0)
                    print(f"[GEE] Downloaded array shape: {array.shape}")
                    return array
                    
            except requests.exceptions.Timeout:
                print(f"[GEE] Timeout on attempt {attempt + 1}")
                if attempt < 2:
                    time.sleep(2)
                    continue
            except Exception as e:
                print(f"[GEE] Download error on attempt {attempt + 1}: {e}")
                if attempt < 2:
                    time.sleep(2)
                    continue
        
        return None
            
    except Exception as e:
        print(f"[GEE] Download error: {e}")
        return None


# ─────────────────────────────────────────────
# MAIN FETCH FUNCTIONS
# ─────────────────────────────────────────────

def fetch_image_as_array(lat, lon, year, buffer_km=5.0, sensor=None):
    """
    Fetch the best available image as a numpy array.
    Returns (array, meta) on success, None on failure.
    """
    # Get the best image
    image, cfg, region = get_best_image(lat, lon, year, buffer_km, sensor)
    
    if image is None:
        return None
    
    # Apply scaling and clip
    image = apply_scaling(image, cfg).clip(region)
    
    # Calculate optimal scale
    side_m = buffer_km * 2 * 1000
    optimal_scale = max(int(side_m / 400), cfg["resolution"])
    optimal_scale = min(optimal_scale, 100)  # Cap at 100m
    
    # Download the actual data
    array = download_image_as_array(image, region, cfg, scale=optimal_scale)
    
    if array is None:
        return None
    
    meta = {
        "sensor": sensor or auto_sensor(year),
        "year": year,
        "source": "GEE",
        "n_bands": array.shape[0],
        "scale_used": optimal_scale,
    }
    
    return array, meta


def fetch_preview_only(lat, lon, year, buffer_km=5.0, sensor=None):
    """Get only a preview URL (much faster)."""
    image, cfg, region = get_best_image(lat, lon, year, buffer_km, sensor)
    
    if image is None:
        return None, None
    
    image = apply_scaling(image, cfg).clip(region)
    
    vis_params = {
        "min": 0,
        "max": 0.3,
        "bands": cfg["bands"][:3]
    }
    
    url = get_preview_url(image, region, vis_params)
    
    meta = {
        "sensor": sensor or auto_sensor(year),
        "year": year,
        "mode": "preview"
    }
    
    return url, meta


def save_array_to_geotiff(array, meta, output_path):
    """Save numpy array to GeoTIFF."""
    try:
        import rasterio
        from rasterio.transform import from_origin
        
        transform = from_origin(0, 0, 1, 1)
        
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=array.shape[1],
            width=array.shape[2],
            count=array.shape[0],
            dtype=np.float32,
            crs='EPSG:4326',
            transform=transform
        ) as dst:
            dst.write(array)
        return output_path
    except Exception as e:
        print(f"[GEE] Save failed: {e}")
        return None


def fetch_and_save(lat, lon, year, buffer_km=5.0, sensor=None):
    """Fetch image and save to temporary file."""
    result = fetch_image_as_array(lat, lon, year, buffer_km, sensor)
    if result is None:
        return None
    
    array, meta = result
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
        path = tmp.name
    
    return save_array_to_geotiff(array, meta, path)
