"""
gee_connector.py — Google Earth Engine direct image fetcher
============================================================
Strategy order:
  1. ee.data.computePixels  — direct in-memory GeoTIFF, no zip, no size limit issues
  2. getDownloadURL (zip)   — fallback, tried at 3 progressively coarser scales
Full error text is printed at every failure point for Streamlit Cloud log debugging.
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
    Initialise Earth Engine via Streamlit service-account secrets,
    falling back to application-default credentials.
    Returns True on success, False on failure.
    """
    try:
        import ee
    except ImportError:
        print("[GEE] earthengine-api not installed")
        return False

    # ── Streamlit secrets path ────────────────────────────────
    try:
        import streamlit as st
        sa_info = st.secrets.get("GEE_SERVICE_ACCOUNT", None)
        if sa_info:
            private_key = sa_info.get("private_key", "")
            private_key = private_key.replace("\\n", "\n")

            if "BEGIN PRIVATE KEY" not in private_key:
                print(f"[GEE] private_key malformed: {private_key[:80]!r}")
                return False

            sa_dict = {
                "type":                        "service_account",
                "project_id":                  sa_info["project_id"],
                "private_key_id":              sa_info["private_key_id"],
                "private_key":                 private_key,
                "client_email":                sa_info["client_email"],
                "client_id":                   sa_info["client_id"],
                "auth_uri":                    "https://accounts.google.com/o/oauth2/auth",
                "token_uri":                   "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url":        sa_info.get("client_x509_cert_url", ""),
            }

            credentials = ee.ServiceAccountCredentials(
                email=sa_dict["client_email"],
                key_data=json.dumps(sa_dict),
            )
            ee.Initialize(
                credentials=credentials,
                project=sa_dict["project_id"],
                opt_url="https://earthengine.googleapis.com",
            )
            test = ee.Number(42).getInfo()
            if test == 42:
                print("[GEE] Initialised via Streamlit service account ✓")
                return True
            print("[GEE] Init appeared to succeed but verification returned unexpected value")
            return False
    except Exception as e:
        print(f"[GEE] Streamlit secrets auth failed: {e}")

    # ── Application-default credentials fallback ──────────────
    try:
        import ee
        ee.Initialize()
        if ee.Number(1).getInfo() == 1:
            print("[GEE] Initialised via application-default credentials ✓")
            return True
    except Exception as e:
        print(f"[GEE] Default credential auth failed: {e}")

    return False


def gee_available() -> bool:
    try:
        import ee
        return ee.Number(1).getInfo() == 1
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
        "offset":       0.0,
        "resolution":   10,
        "cloud_prop":   "CLOUDY_PIXEL_PERCENTAGE",
        "cloud_max":    30,
        "start_year":   2017,
    },
    "Sentinel-2 L2A (old)": {
        "collection":   "COPERNICUS/S2_SR",
        "bands":        ["B2", "B3", "B4", "B8", "B11", "B12"],
        "scale_factor": 0.0001,
        "offset":       0.0,
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
    for name in _SENSOR_PRIORITY:
        if year >= SENSOR_CONFIGS[name]["start_year"]:
            return name
    return "Landsat 5 TM"


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _build_collection(cfg, aoi, year, month_start, month_end, cloud_max):
    import ee
    start = f"{year}-{month_start:02d}-01"
    end   = f"{year + 1}-01-01" if month_end == 12 else f"{year}-{month_end + 1:02d}-01"
    return (
        ee.ImageCollection(cfg["collection"])
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt(cfg["cloud_prop"], cloud_max))
        .select(cfg["bands"])
    )


def _apply_scaling(array: np.ndarray, cfg: dict) -> np.ndarray:
    """Convert raw DN → surface reflectance and clip to [0, 1]."""
    array = array.astype(np.float32) * cfg["scale_factor"] + cfg.get("offset", 0.0)
    return np.clip(array, 0.0, 1.0)


def _scale_for_aoi(buffer_km: float, native_res: int,
                   target_px: int = 256) -> int:
    """Choose a download scale that gives ~target_px per side."""
    side_m = buffer_km * 2 * 1000
    scale  = max(int(side_m / target_px), native_res, 30)
    return int(np.ceil(scale / 10.0) * 10)


# ─────────────────────────────────────────────────────────────
# DOWNLOAD STRATEGY 1 — computePixels  (preferred)
# ─────────────────────────────────────────────────────────────

def _try_compute_pixels(image, aoi, cfg: dict, scale: int):
    """
    Download via ee.data.computePixels — returns a raw GeoTIFF
    blob directly without a size-limited zip.
    Requires earthengine-api >= 0.1.374.
    Returns (C,H,W) float32 array or None.
    """
    try:
        import ee
        import rasterio
        from rasterio.io import MemoryFile

        print(f"[GEE] Trying computePixels at {scale}m …")

        pixel_grid = {
            "crsCode": "EPSG:4326",
            "scale": {"xScale": scale / 111320.0,
                      "yScale": scale / 111320.0},
        }

        params = {
            "expression": image.clip(aoi),
            "fileFormat": "GEO_TIFF",
            "bandIds":    cfg["bands"],
            "grid":       pixel_grid,
            "region":     aoi.getInfo()["coordinates"],
        }

        raw = ee.data.computePixels(params)

        with MemoryFile(raw) as mf:
            with mf.open() as ds:
                array = ds.read().astype(np.float32)

        array = _apply_scaling(array, cfg)
        print(f"[GEE] computePixels OK — shape={array.shape} "
              f"range=[{array.min():.3f}, {array.max():.3f}]")
        return array

    except Exception as e:
        print(f"[GEE] computePixels failed: {type(e).__name__}: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# DOWNLOAD STRATEGY 2 — getDownloadURL + zip  (fallback)
# ─────────────────────────────────────────────────────────────

def _try_download_url(image, aoi, cfg: dict, scale: int):
    """
    Download via getDownloadURL (per-band zip).
    Returns (C,H,W) float32 array or None.
    """
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

    print(f"[GEE] Trying getDownloadURL at {scale}m …")
    try:
        url = image.getDownloadURL(params)
    except Exception as e:
        print(f"[GEE] getDownloadURL RPC failed: {type(e).__name__}: {e}")
        return None

    try:
        r = requests.get(url, timeout=360)
    except Exception as e:
        print(f"[GEE] HTTP request failed: {type(e).__name__}: {e}")
        return None

    ct      = r.headers.get("Content-Type", "")
    size_kb = len(r.content) / 1024
    print(f"[GEE] Response: HTTP {r.status_code} | {size_kb:.1f} KB | {ct}")

    if r.status_code != 200:
        print(f"[GEE] Non-200 body: {r.content[:800].decode('utf-8', errors='replace')}")
        return None

    # GEE error responses come back as JSON or HTML even with HTTP 200
    first_byte = r.content[:1]
    if first_byte in (b"{", b"<") or "json" in ct or "html" in ct:
        print(f"[GEE] GEE returned error (not a zip): "
              f"{r.content[:800].decode('utf-8', errors='replace')}")
        return None

    if len(r.content) < 4 or r.content[:2] != b"PK":
        print(f"[GEE] Not a zip (bad magic): {r.content[:200]!r}")
        return None

    bands = []
    try:
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            tifs = sorted(n for n in z.namelist() if n.lower().endswith(".tif"))
            print(f"[GEE] ZIP: {len(tifs)} TIF(s): {tifs}")
            if not tifs:
                print("[GEE] Zip has no .tif files")
                return None
            for name in tifs:
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
    except zipfile.BadZipFile:
        print(f"[GEE] BadZipFile: {r.content[:400]!r}")
        return None
    except Exception as e:
        import traceback
        print(f"[GEE] Zip extraction error: {e}")
        traceback.print_exc()
        return None

    if not bands:
        print("[GEE] No bands extracted from zip")
        return None

    array = np.stack(bands, axis=0)
    array = _apply_scaling(array, cfg)
    print(f"[GEE] getDownloadURL OK — shape={array.shape} "
          f"range=[{array.min():.3f}, {array.max():.3f}]")
    return array


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
):
    """
    Fetch a cloud-free median composite for a location/year.
    Returns (array, meta) on success, None on failure.
    array is (C, H, W) float32 in [0, 1].
    """
    try:
        import ee
    except ImportError:
        print("[GEE] earthengine-api not installed")
        return None

    sensor = sensor or auto_select_sensor(year)
    cfg    = SENSOR_CONFIGS[sensor]
    print(f"[GEE] ── fetch start ─────────────────────────────")
    print(f"[GEE] sensor={sensor}  year={year}  "
          f"lat={lat:.4f}  lon={lon:.4f}  buffer={buffer_km}km")

    aoi = ee.Geometry.Point([lon, lat]).buffer(buffer_km * 1000).bounds()

    # ── Find scenes with progressive cloud relaxation ─────────
    collection = None
    for cloud_limit in [cfg["cloud_max"], 50, 80]:
        col   = _build_collection(cfg, aoi, year, month_start, month_end, cloud_limit)
        count = col.size().getInfo()
        print(f"[GEE] Scenes cloud<{cloud_limit}%: {count}")
        if count > 0:
            collection = col
            break

    # Full-year last resort
    if collection is None:
        col   = _build_collection(cfg, aoi, year, 1, 12, 80)
        count = col.size().getInfo()
        print(f"[GEE] Full-year scenes cloud<80%: {count}")
        if count == 0:
            # Try Landsat fallback if S2 had nothing
            if "Sentinel" in sensor:
                fallback = "Landsat 8/9" if year >= 2013 else "Landsat 5 TM"
                print(f"[GEE] S2 empty — falling back to {fallback}")
                return fetch_image_as_array(
                    lat=lat, lon=lon, year=year,
                    month_start=month_start, month_end=month_end,
                    sensor=fallback, buffer_km=buffer_km,
                )
            print("[GEE] No scenes found — giving up")
            return None
        collection = col

    # ── Build scaled median composite ────────────────────────
    image = collection.median().clip(aoi)

    # ── Determine download scales to try ─────────────────────
    base_scale   = _scale_for_aoi(buffer_km, cfg["resolution"], target_px=256)
    coarse_scale = max(base_scale * 4, 300)
    scales       = sorted(set([base_scale, coarse_scale, 1000]))
    print(f"[GEE] Will try scales: {scales}m")

    # ── Strategy 1: computePixels ─────────────────────────────
    array = _try_compute_pixels(image, aoi, cfg, base_scale)

    # ── Strategy 2: getDownloadURL at each scale ──────────────
    if array is None:
        for scale in scales:
            print(f"[GEE] Trying getDownloadURL at {scale}m …")
            array = _try_download_url(image, aoi, cfg, scale)
            if array is not None:
                break

    if array is None:
        print("[GEE] ── ALL download strategies failed ─────────")
        return None

    meta = {
        "sensor":  sensor,
        "year":    year,
        "source":  "GEE",
        "n_bands": array.shape[0],
    }
    print(f"[GEE] ── fetch complete ✓ ───────────────────────")
    return array, meta


# ─────────────────────────────────────────────────────────────
# SAVE HELPER
# ─────────────────────────────────────────────────────────────

def save_array_as_geotiff(array: np.ndarray, meta: dict, output_path: str):
    """Write a (C, H, W) float32 array to a GeoTIFF file."""
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
):
    """Fetch an image and save it to a temporary GeoTIFF. Returns path or None."""
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
