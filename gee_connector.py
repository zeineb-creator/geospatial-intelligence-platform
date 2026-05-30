"""
gee_connector.py — Google Earth Engine direct image fetcher
============================================================
Robust implementation with proper error handling and fallback strategies.
"""

import os
import io
import json
import tempfile
import zipfile
import numpy as np
import time
import ee


# ─────────────────────────────────────────────────────────────
# GEE INITIALISATION
# ─────────────────────────────────────────────────────────────

def init_gee() -> bool:
    """
    Initialise Earth Engine. Tries Streamlit service-account secrets first,
    then falls back to application-default credentials.
    Returns True on success, False on failure.
    """
    try:
        import ee
    except ImportError:
        print("[GEE] earthengine-api not installed")
        return False

    # ── Try Streamlit secrets ─────────────────────────────────
    try:
        import streamlit as st
        sa_info = st.secrets.get("GEE_SERVICE_ACCOUNT", None)
        if sa_info:
            sa_dict = {
                "type":                        sa_info.get("type", "service_account"),
                "project_id":                  sa_info["project_id"],
                "private_key_id":              sa_info["private_key_id"],
                "private_key":                 sa_info["private_key"].replace("\\n", "\n"),
                "client_email":                sa_info["client_email"],
                "client_id":                   sa_info["client_id"],
                "auth_uri":                    sa_info.get("auth_uri",  "https://accounts.google.com/o/oauth2/auth"),
                "token_uri":                   sa_info.get("token_uri", "https://oauth2.googleapis.com/token"),
                "auth_provider_x509_cert_url": sa_info.get("auth_provider_x509_cert_url",
                                                            "https://www.googleapis.com/oauth2/v1/certs"),
                "client_x509_cert_url":        sa_info.get("client_x509_cert_url", ""),
            }
            credentials = ee.ServiceAccountCredentials(
                email=sa_dict["client_email"],
                key_data=json.dumps(sa_dict),
            )
            ee.Initialize(credentials, project=sa_dict["project_id"])
            print("[GEE] Initialised via Streamlit service account")
            return True
    except Exception as e:
        print(f"[GEE] Streamlit secrets auth failed: {e}")

    # ── Try application-default credentials ───────────────────
    try:
        import ee
        ee.Initialize()
        print("[GEE] Initialised via application-default credentials")
        return True
    except Exception as e:
        print(f"[GEE] Default credential auth failed: {e}")

    return False


def gee_available() -> bool:
    """Quick liveness check — returns True if GEE can execute a computation."""
    try:
        import ee
        ee.Number(1).getInfo()
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────
# SENSOR CONFIGURATIONS
# ─────────────────────────────────────────────────────────────

SENSOR_CONFIGS = {
    "Sentinel-2 L2A": {
        "collection":   "COPERNICUS/S2_SR_HARMONIZED",
        "bands":        ["B2", "B3", "B4", "B8", "B11", "B12"],
        "scale_factor": 0.0001,
        "resolution":   10,
        "cloud_prop":   "CLOUDY_PIXEL_PERCENTAGE",
        "cloud_max":    30,
        "start_year":   2017,
    },
    "Sentinel-2 L2A (old)": {
        "collection":   "COPERNICUS/S2_SR",
        "bands":        ["B2", "B3", "B4", "B8", "B11", "B12"],
        "scale_factor": 0.0001,
        "resolution":   10,
        "cloud_prop":   "CLOUDY_PIXEL_PERCENTAGE",
        "cloud_max":    30,
        "start_year":   2015,
    },
    "Landsat 8/9": {
        "collection":   "LANDSAT/LC08/C02/T1_L2",
        "bands":        ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
        "scale_factor": 0.0000275,
        "offset":       -0.2,
        "resolution":   30,
        "cloud_prop":   "CLOUD_COVER",
        "cloud_max":    30,
        "start_year":   2013,
    },
    "Landsat 7": {
        "collection":   "LANDSAT/LE07/C02/T1_L2",
        "bands":        ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
        "scale_factor": 0.0000275,
        "offset":       -0.2,
        "resolution":   30,
        "cloud_prop":   "CLOUD_COVER",
        "cloud_max":    30,
        "start_year":   1999,
    },
    "Landsat 5 TM": {
        "collection":   "LANDSAT/LT05/C02/T1_L2",
        "bands":        ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
        "scale_factor": 0.0000275,
        "offset":       -0.2,
        "resolution":   30,
        "cloud_prop":   "CLOUD_COVER",
        "cloud_max":    30,
        "start_year":   1984,
    },
}

_SENSOR_PRIORITY = [
    "Sentinel-2 L2A",
    "Sentinel-2 L2A (old)",
    "Landsat 8/9",
    "Landsat 7",
    "Landsat 5 TM",
]


def auto_select_sensor(year: int) -> str:
    """Return the best sensor available for a given year."""
    for name in _SENSOR_PRIORITY:
        cfg = SENSOR_CONFIGS[name]
        if year >= cfg["start_year"]:
            return name
    return "Landsat 5 TM"


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _aoi_from_point(lon: float, lat: float, buffer_km: float):
    """Return a GEE rectangular AOI buffered around a point."""
    return ee.Geometry.Point([lon, lat]).buffer(buffer_km * 1000).bounds()


def _build_collection(cfg: dict, aoi, year: int,
                      month_start: int = 1, month_end: int = 12,
                      cloud_max_override: float = None):
    """Build a filtered, band-selected ImageCollection."""
    cloud_max = cloud_max_override if cloud_max_override is not None else cfg["cloud_max"]
    start = f"{year}-{month_start:02d}-01"
    if month_end == 12:
        end = f"{year + 1}-01-01"
    else:
        end = f"{year}-{month_end + 1:02d}-01"

    return (
        ee.ImageCollection(cfg["collection"])
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt(cfg["cloud_prop"], cloud_max))
        .select(cfg["bands"])
    )


def _apply_scaling(image, cfg: dict):
    """Apply scale + optional offset to convert to surface reflectance [0, 1]."""
    image = image.multiply(cfg["scale_factor"])
    if "offset" in cfg:
        image = image.add(cfg["offset"])
    return image


def _optimal_download_scale(buffer_km: float, native_res: int,
                             target_pixels: int = 400) -> int:
    """
    Compute a download resolution that keeps the raster size manageable.
    """
    side_m = buffer_km * 2 * 1000
    raw_scale = int(side_m / target_pixels)
    scale = max(raw_scale, native_res)
    scale = max(scale, 30)
    scale = int(np.ceil(scale / 10.0) * 10)
    print(f"[GEE] AOI side={side_m:.0f}m → download scale={scale}m")
    return scale


# ─────────────────────────────────────────────────────────────
# DOWNLOAD STRATEGY — getDownloadURL with robust error handling
# ─────────────────────────────────────────────────────────────

def _download_via_url(image, aoi, cfg: dict, scale: int) -> np.ndarray | None:
    """
    Download image using getDownloadURL with improved error handling.
    Returns (C, H, W) float32 array clipped to [0, 1], or None on failure.
    """
    try:
        import requests
        import rasterio

        # Ensure the image is properly clipped to AOI
        clipped = image.clip(aoi)
        
        # Get the AOI bounds as a GeoJSON polygon
        try:
            # Get bounds from AOI
            bounds_dict = aoi.bounds().getInfo()
            
            # The bounds are returned as a GeoJSON polygon
            # Extract the coordinates properly
            if 'coordinates' in bounds_dict and len(bounds_dict['coordinates']) > 0:
                coords = bounds_dict['coordinates'][0]
                # Extract min/max from the coordinates
                min_x = min(p[0] for p in coords)
                max_x = max(p[0] for p in coords)
                min_y = min(p[1] for p in coords)
                max_y = max(p[1] for p in coords)
            else:
                # Fallback: get coordinates directly
                coords_list = aoi.coordinates().getInfo()
                if isinstance(coords_list, list) and len(coords_list) > 0:
                    if len(coords_list[0]) > 0:
                        points = coords_list[0]
                        min_x = min(p[0] for p in points)
                        max_x = max(p[0] for p in points)
                        min_y = min(p[1] for p in points)
                        max_y = max(p[1] for p in points)
                    else:
                        raise ValueError("Empty coordinates")
                else:
                    raise ValueError("Invalid coordinates format")
            
            print(f"[GEE] AOI bounds: ({min_x:.4f}, {min_y:.4f}) to ({max_x:.4f}, {max_y:.4f})")
            
        except Exception as e:
            print(f"[GEE] Error getting AOI bounds: {e}")
            # Last resort: get the centroid and create a square
            try:
                centroid = aoi.centroid().coordinates().getInfo()
                center_lon, center_lat = centroid[0], centroid[1]
                # Approximate degrees for the buffer size
                half_side_deg = (buffer_km * 1000) / 111320  # ~111 km per degree at equator
                min_x = center_lon - half_side_deg
                max_x = center_lon + half_side_deg
                min_y = center_lat - half_side_deg
                max_y = center_lat + half_side_deg
                print(f"[GEE] Using fallback bounds around ({center_lon:.4f}, {center_lat:.4f})")
            except:
                print("[GEE] Cannot determine AOI bounds for download")
                return None
        
        # Create GeoJSON polygon for the region
        region = {
            "type": "Polygon",
            "coordinates": [[
                [min_x, min_y],
                [min_x, max_y],
                [max_x, max_y],
                [max_x, min_y],
                [min_x, min_y]
            ]]
        }
        
        # Build download parameters
        params = {
            "scale": scale,
            "region": region,
            "format": "GEO_TIFF",
            "crs": "EPSG:4326",
        }
        
        print(f"[GEE] Requesting download at {scale}m scale...")
        url = clipped.getDownloadURL(params)
        print(f"[GEE] Download URL obtained")
        
        # Download with retry logic
        max_retries = 2
        for attempt in range(max_retries):
            try:
                r = requests.get(url, timeout=300)
                
                if r.status_code != 200:
                    print(f"[GEE] HTTP {r.status_code}")
                    if r.text:
                        print(f"[GEE] Error response: {r.text[:500]}")
                    return None
                
                # Check if response is valid
                if not r.content or len(r.content) < 100:
                    print(f"[GEE] Empty or very small response")
                    return None
                
                # Check for zip file (PK header)
                if r.content[:2] != b'PK':
                    # Try to decode as text for error message
                    try:
                        preview = r.content[:200].decode('utf-8', errors='ignore')
                        if 'error' in preview.lower():
                            print(f"[GEE] Error response: {preview}")
                        else:
                            print(f"[GEE] Response is not a zip file. Preview: {preview[:100]}")
                    except:
                        print("[GEE] Response is not a zip file")
                    return None
                
                print(f"[GEE] Download successful, size: {len(r.content)} bytes")
                
                # Process zip file
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    # Find TIF files
                    tif_names = [n for n in z.namelist() if n.lower().endswith('.tif')]
                    if not tif_names:
                        print("[GEE] No TIF files found in zip")
                        return None
                    
                    print(f"[GEE] Found {len(tif_names)} TIF file(s): {tif_names}")
                    
                    # Read all bands
                    bands = []
                    for name in tif_names:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
                            tmp.write(z.read(name))
                            tmp_path = tmp.name
                        
                        try:
                            with rasterio.open(tmp_path) as src:
                                band_data = src.read(1).astype(np.float32)
                                bands.append(band_data)
                        finally:
                            try:
                                os.unlink(tmp_path)
                            except OSError:
                                pass
                    
                    if not bands:
                        print("[GEE] No bands extracted")
                        return None
                    
                    # Stack bands
                    if len(bands) == 1:
                        array = bands[0].reshape(1, bands[0].shape[0], bands[0].shape[1])
                    else:
                        array = np.stack(bands, axis=0)
                    
                    array = np.clip(array, 0.0, 1.0)
                    print(f"[GEE] Success: array shape {array.shape}")
                    return array
                
            except requests.exceptions.Timeout:
                print(f"[GEE] Timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
            except zipfile.BadZipFile as e:
                print(f"[GEE] BadZipFile: {e}")
                return None
            except Exception as e:
                print(f"[GEE] Error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
        
        return None
        
    except Exception as e:
        import traceback
        print(f"[GEE] Unexpected error: {e}")
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────
# MAIN FETCH FUNCTION
# ─────────────────────────────────────────────────────────────

def fetch_image_as_array(
    lat: float,
    lon: float,
    year: int,
    month_start: int = 1,
    month_end: int = 12,
    sensor: str = None,
    buffer_km: float = 5.0,
) -> tuple | None:
    """
    Fetch a cloud-free median composite for a location / year.
    Returns (array, meta) on success, None on failure.
    """
    try:
        import ee
    except ImportError:
        print("[GEE] earthengine-api not installed")
        return None

    # ── Sensor selection ──────────────────────────────────────
    sensor = sensor or auto_select_sensor(year)
    cfg = SENSOR_CONFIGS[sensor]
    print(f"[GEE] Sensor={sensor} year={year} lat={lat:.4f} lon={lon:.4f} buffer={buffer_km}km")

    # ── AOI ───────────────────────────────────────────────────
    aoi = _aoi_from_point(lon, lat, buffer_km)

    # ── Build collection with progressive cloud relaxation ────
    collection = None
    cloud_limits = [cfg["cloud_max"], 50, 80]
    count = 0
    
    for cloud_limit in cloud_limits:
        try:
            col = _build_collection(cfg, aoi, year, month_start, month_end, cloud_max_override=cloud_limit)
            count = col.size().getInfo()
            print(f"[GEE] Scenes (cloud<{cloud_limit}%): {count}")
            if count > 0:
                collection = col
                break
        except Exception as e:
            print(f"[GEE] Error at cloud limit {cloud_limit}: {e}")
            continue

    if collection is None or count == 0:
        # Last resort: full year with high cloud tolerance
        try:
            col = _build_collection(cfg, aoi, year, 1, 12, cloud_max_override=80)
            count = col.size().getInfo()
            print(f"[GEE] Full-year fallback scenes: {count}")
            if count == 0:
                print(f"[GEE] No scenes found for {sensor} {year}")
                return None
            collection = col
        except Exception as e:
            print(f"[GEE] Fallback failed: {e}")
            return None

    # ── Build composite and scale ─────────────────────────────
    try:
        composite = collection.median()
        image = _apply_scaling(composite, cfg).clip(aoi)
    except Exception as e:
        print(f"[GEE] Error building composite: {e}")
        return None

    # ── Download with progressive scale attempts ──────────────
    scales_to_try = [
        _optimal_download_scale(buffer_km, cfg["resolution"], target_pixels=400),
        _optimal_download_scale(buffer_km, cfg["resolution"], target_pixels=300),
        _optimal_download_scale(buffer_km, cfg["resolution"], target_pixels=200),
        500
    ]
    scales_to_try = list(set(scales_to_try))  # Remove duplicates
    
    array = None
    final_scale = None
    
    for try_scale in scales_to_try:
        print(f"[GEE] Trying scale: {try_scale}m")
        array = _download_via_url(image, aoi, cfg, try_scale)
        if array is not None:
            final_scale = try_scale
            break
        time.sleep(1)
    
    if array is None:
        print("[GEE] All download attempts failed")
        return None

    meta = {
        "sensor": sensor,
        "year": year,
        "source": "GEE",
        "n_bands": array.shape[0],
        "scale_used": final_scale,
    }
    print(f"[GEE] ✓ Success: shape={array.shape}, range=[{array.min():.3f}, {array.max():.3f}]")
    return array, meta


# ─────────────────────────────────────────────────────────────
# SAVE HELPER
# ─────────────────────────────────────────────────────────────

def save_array_as_geotiff(array: np.ndarray, meta: dict, output_path: str) -> str | None:
    """Write a (C, H, W) float32 array to a GeoTIFF at output_path."""
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


def fetch_and_save(
    lat: float,
    lon: float,
    year: int,
    month_start: int = 1,
    month_end: int = 12,
    sensor: str = None,
    buffer_km: float = 5.0,
) -> str | None:
    """Convenience wrapper: fetch an image and save it to a temp GeoTIFF."""
    result = fetch_image_as_array(
        lat=lat, lon=lon, year=year,
        month_start=month_start, month_end=month_end,
        sensor=sensor, buffer_km=buffer_km,
    )
    if result is None:
        return None

    array, meta = result
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        path = tmp.name

    return save_array_as_geotiff(array, meta, path)
