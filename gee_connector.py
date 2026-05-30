"""
gee_connector.py — Google Earth Engine direct image fetcher
============================================================
Robust implementation with:
  - Multi-strategy download (computePixels → getDownloadURL fallback)
  - Correct sensor auto-selection with collection date guards
  - Adaptive resolution scaling to stay under GEE memory limits
  - Full error surfacing so failures are diagnosable
  - Both Sentinel-2 SR collections (older + harmonised)
"""

import os
import io
import json
import tempfile
import zipfile
import numpy as np


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
# Each entry defines:
#   collection   – GEE ImageCollection ID
#   bands        – band names to select (exported in this order: B/G/R/NIR/SWIR1/SWIR2)
#   scale_factor – multiplicative reflectance scaling
#   offset       – additive offset (Landsat C02 only)
#   resolution   – native GSD in metres
#   cloud_prop   – property name used for cloud filtering
#   cloud_max    – maximum cloud % allowed in initial filter
#   start_year   – first year data are available

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
        # Pre-harmonised collection; covers 2015–2016 reprocessed scenes
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

# Ordered preference list used by auto_select_sensor
_SENSOR_PRIORITY = [
    "Sentinel-2 L2A",
    "Sentinel-2 L2A (old)",
    "Landsat 8/9",
    "Landsat 7",
    "Landsat 5 TM",
]


def auto_select_sensor(year: int) -> str:
    """
    Return the best sensor available for a given year.
    Prefers highest-resolution option; falls back through the priority list.
    """
    for name in _SENSOR_PRIORITY:
        cfg = SENSOR_CONFIGS[name]
        if year >= cfg["start_year"]:
            return name
    # Absolute fallback
    return "Landsat 5 TM"


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _aoi_from_point(lon: float, lat: float, buffer_km: float):
    """Return a GEE rectangular AOI buffered around a point."""
    import ee
    return ee.Geometry.Point([lon, lat]).buffer(buffer_km * 1000).bounds()


def _build_collection(cfg: dict, aoi, year: int,
                      month_start: int = 1, month_end: int = 12,
                      cloud_max_override: float = None):
    """Build a filtered, band-selected ImageCollection."""
    import ee
    cloud_max = cloud_max_override if cloud_max_override is not None else cfg["cloud_max"]
    start = f"{year}-{month_start:02d}-01"
    # End on the last day of month_end (use first day of next month)
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
    import ee
    image = image.multiply(cfg["scale_factor"])
    if "offset" in cfg:
        image = image.add(cfg["offset"])
    return image


def _optimal_download_scale(buffer_km: float, native_res: int,
                             target_pixels: int = 512) -> int:
    """
    Compute a download resolution that:
      - Keeps the raster below ~target_pixels × target_pixels per band
      - Is never finer than the sensor's native resolution
      - Stays well within GEE's ~32 MB per download limit
    """
    side_m = buffer_km * 2 * 1000          # AOI side length in metres
    raw_scale = int(side_m / target_pixels) # metres per pixel to hit target_pixels
    scale = max(raw_scale, native_res)      # never finer than native
    scale = max(scale, 30)                  # hard floor: 30 m
    # Round up to nearest 10 m for clean numbers
    scale = int(np.ceil(scale / 10.0) * 10)
    print(f"[GEE] AOI side={side_m:.0f}m  native={native_res}m  "
          f"→ download scale={scale}m ({side_m/scale:.0f}px per side)")
    return scale


# ─────────────────────────────────────────────────────────────
# DOWNLOAD STRATEGY 1 — computePixels  (preferred, no zip)
# ─────────────────────────────────────────────────────────────

def _download_via_compute_pixels(image, aoi, cfg: dict,
                                  scale: int) -> np.ndarray | None:
    """
    Use ee.data.computePixels to download a GeoTIFF directly into memory.
    Requires earthengine-api >= 0.1.374.
    Returns (C, H, W) float32 array clipped to [0, 1], or None on failure.
    """
    try:
        import ee
        import rasterio
        from rasterio.io import MemoryFile

        request = {
            "expression": image,
            "fileFormat": "GEO_TIFF",
            "bandIds": cfg["bands"],
            "grid": {
                "dimensions": {"width": 512, "height": 512},
                "affineTransform": {},
                "crsCode": "EPSG:4326",
            },
            "region": aoi,
        }

        # Simpler form that works across more API versions
        params = {
            "expression": image.clip(aoi),
            "fileFormat": "GEO_TIFF",
            "scale": scale,
            "region": aoi.getInfo(),
            "crs": "EPSG:4326",
            "bands": [{"id": b} for b in cfg["bands"]],
        }

        print("[GEE] Trying computePixels…")
        raw = ee.data.computePixels(params)

        with MemoryFile(raw) as mf:
            with mf.open() as ds:
                array = ds.read().astype(np.float32)

        array = np.clip(array, 0.0, 1.0)
        print(f"[GEE] computePixels success — shape {array.shape}")
        return array

    except Exception as e:
        print(f"[GEE] computePixels failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# DOWNLOAD STRATEGY 2 — getDownloadURL + zip  (fallback)
# ─────────────────────────────────────────────────────────────

def _download_via_url(image, aoi, cfg: dict, scale: int) -> np.ndarray | None:
    """
    Use getDownloadURL to retrieve a zip of per-band GeoTIFFs.
    Returns (C, H, W) float32 array clipped to [0, 1], or None on failure.
    """
    try:
        import requests
        import rasterio

        params = {
            "scale":       scale,
            "region":      aoi.getInfo(),
            "format":      "GEO_TIFF",
            "bands":       cfg["bands"],
            "crs":         "EPSG:4326",
            "filePerBand": True,
        }

        print(f"[GEE] Requesting getDownloadURL at {scale}m…")
        url = image.getDownloadURL(params)
        print(f"[GEE] URL obtained, downloading…")

        r = requests.get(url, timeout=360)

        # ── Detailed error surfacing ──────────────────────────
        content_type = r.headers.get("Content-Type", "")
        print(f"[GEE] Response: HTTP {r.status_code} | "
              f"size={len(r.content)/1024:.1f} KB | type={content_type}")

        if r.status_code != 200:
            body = r.content[:800].decode("utf-8", errors="replace")
            print(f"[GEE] HTTP error body: {body}")
            return None

        # If GEE returns JSON it's an error message
        if "json" in content_type or (r.content[:1] == b"{"):
            body = r.content[:800].decode("utf-8", errors="replace")
            print(f"[GEE] GEE returned error JSON: {body}")
            return None

        # Verify it's a zip
        is_zip = ("zip" in content_type or
                  "octet-stream" in content_type or
                  r.content[:2] == b"PK")
        if not is_zip:
            body = r.content[:800].decode("utf-8", errors="replace")
            print(f"[GEE] Not a zip file. Response preview: {body}")
            return None

        # ── Unzip and stack bands ─────────────────────────────
        bands = []
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            tif_names = sorted(n for n in z.namelist() if n.lower().endswith(".tif"))
            print(f"[GEE] ZIP contains {len(tif_names)} TIF(s): {tif_names}")
            if not tif_names:
                print("[GEE] No .tif files found in zip")
                return None

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
                    except OSError:
                        pass

        if not bands:
            print("[GEE] No bands extracted from zip")
            return None

        array = np.stack(bands, axis=0)          # (C, H, W)
        array = np.clip(array, 0.0, 1.0)
        print(f"[GEE] getDownloadURL success — shape {array.shape}")
        return array

    except zipfile.BadZipFile:
        body = getattr(r, "content", b"")[:500].decode("utf-8", errors="replace")
        print(f"[GEE] BadZipFile — likely a GEE quota/size error. Body: {body}")
        return None
    except Exception as e:
        import traceback
        print(f"[GEE] getDownloadURL error: {e}")
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
    Fetch a cloud-free median composite for a location / year and return it as
    a (C, H, W) float32 numpy array with values in [0, 1].

    Parameters
    ----------
    lat, lon     : Centre of the AOI
    year         : Image acquisition year
    month_start  : Start month of the compositing window (default 1)
    month_end    : End month of the compositing window (default 12)
    sensor       : Force a specific sensor; None = auto-select
    buffer_km    : Half-width of the square AOI in km

    Returns
    -------
    (array, meta) on success, None on failure.
    meta contains: sensor, year, source, n_bands, scale_used
    """
    try:
        import ee
    except ImportError:
        print("[GEE] earthengine-api not installed")
        return None

    # ── Sensor selection ──────────────────────────────────────
    sensor = sensor or auto_select_sensor(year)
    cfg    = SENSOR_CONFIGS[sensor]
    print(f"[GEE] Sensor={sensor}  year={year}  "
          f"lat={lat:.4f}  lon={lon:.4f}  buffer={buffer_km}km")

    # ── AOI ───────────────────────────────────────────────────
    aoi = _aoi_from_point(lon, lat, buffer_km)

    # ── Build collection with progressive cloud relaxation ────
    collection = None
    cloud_limits = [cfg["cloud_max"], 50, 80]
    for cloud_limit in cloud_limits:
        col = _build_collection(cfg, aoi, year,
                                month_start, month_end,
                                cloud_max_override=cloud_limit)
        count = col.size().getInfo()
        print(f"[GEE] Scenes (cloud<{cloud_limit}%): {count}")
        if count > 0:
            collection = col
            break

    if collection is None or count == 0:
        # Last resort: try full year with cloud<80%
        col = _build_collection(cfg, aoi, year, 1, 12, cloud_max_override=80)
        count = col.size().getInfo()
        print(f"[GEE] Full-year fallback scenes (cloud<80%): {count}")
        if count == 0:
            print(f"[GEE] No scenes found for {sensor} {year} — "
                  "try a different year, larger buffer, or different sensor")
            return None
        collection = col

    # ── Build composite and scale ─────────────────────────────
    image = _apply_scaling(collection.median(), cfg).clip(aoi)

    # ── Adaptive download resolution ─────────────────────────
    scale = _optimal_download_scale(buffer_km, cfg["resolution"])

    # ── Strategy 1: computePixels ─────────────────────────────
    array = _download_via_compute_pixels(image, aoi, cfg, scale)

    # ── Strategy 2: getDownloadURL fallback ───────────────────
    if array is None:
        print("[GEE] Falling back to getDownloadURL strategy…")
        array = _download_via_url(image, aoi, cfg, scale)

    # ── Strategy 3: coarser scale retry ──────────────────────
    if array is None and scale < 500:
        coarser = max(scale * 4, 500)
        print(f"[GEE] Retrying at coarser scale {coarser}m…")
        array = _download_via_url(image, aoi, cfg, coarser)
        if array is not None:
            scale = coarser

    if array is None:
        print("[GEE] All download strategies failed")
        return None

    meta = {
        "sensor":   sensor,
        "year":     year,
        "source":   "GEE",
        "n_bands":  array.shape[0],
        "scale_used": scale,
    }
    print(f"[GEE] ✓ array shape={array.shape}  "
          f"range=[{array.min():.3f}, {array.max():.3f}]")
    return array, meta


# ─────────────────────────────────────────────────────────────
# SAVE HELPER
# ─────────────────────────────────────────────────────────────

def save_array_as_geotiff(array: np.ndarray, meta: dict,
                           output_path: str) -> str | None:
    """Write a (C, H, W) float32 array to a GeoTIFF at output_path."""
    try:
        import rasterio
        with rasterio.open(
            output_path, "w",
            driver="GTiff",
            height=array.shape[1],
            width=array.shape[2],
            count=array.shape[0],
            dtype=np.float32,
        ) as dst:
            dst.write(array)
        return output_path
    except Exception as e:
        print(f"[GEE] save_array_as_geotiff failed: {e}")
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
    """
    Convenience wrapper: fetch an image and save it to a temp GeoTIFF.
    Returns the file path on success, None on failure.
    """
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
