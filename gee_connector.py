"""
gee_connector.py — Google Earth Engine direct image fetcher
============================================================
Robust implementation with:
  - Fixed init_gee() syntax (missing return True + except block)
  - Multi-strategy download (getDownloadURL with adaptive scale + coarser retry)
  - Correct sensor auto-selection with year guards
  - Progressive cloud relaxation (30% → 50% → 80%)
  - Full error surfacing so failures are diagnosable in Streamlit logs
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
    Initialise Earth Engine.
    Tries Streamlit service-account secrets first,
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
        if not sa_info:
            print("[GEE] No GEE_SERVICE_ACCOUNT found in secrets")
        else:
            print(f"[GEE] client_email = {sa_info.get('client_email', 'MISSING')}")
            print(f"[GEE] project_id   = {sa_info.get('project_id',   'MISSING')}")

            private_key = sa_info.get("private_key", "")
            # Fix newlines — Streamlit TOML may store literal \n
            private_key = private_key.replace("\\n", "\n")

            if "BEGIN PRIVATE KEY" not in private_key:
                print(f"[GEE] private_key looks malformed — first 80 chars: {private_key[:80]!r}")
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

            # Verify it actually works
            test = ee.Number(42).getInfo()
            print(f"[GEE] Verification: {test}")
            if test == 42:
                print("[GEE] Initialised via Streamlit service account ✓")
                return True
            else:
                print("[GEE] Verification returned unexpected value")
                return False

    except Exception as e:
        print(f"[GEE] Streamlit secrets auth failed: {e}")

    # ── Fallback: application-default credentials ─────────────
    try:
        import ee
        ee.Initialize()
        test = ee.Number(1).getInfo()
        if test == 1:
            print("[GEE] Initialised via application-default credentials ✓")
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
        # Pre-harmonised collection; covers some reprocessed 2015-2016 scenes
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

# Priority order used by auto_select_sensor
_SENSOR_PRIORITY = [
    "Sentinel-2 L2A",
    "Sentinel-2 L2A (old)",
    "Landsat 8/9",
    "Landsat 7",
    "Landsat 5 TM",
]


def auto_select_sensor(year: int) -> str:
    """Return the best available sensor for a given year."""
    for name in _SENSOR_PRIORITY:
        if year >= SENSOR_CONFIGS[name]["start_year"]:
            return name
    return "Landsat 5 TM"


# ─────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────

def _optimal_scale(buffer_km: float, native_res: int) -> int:
    """
    Pick a download resolution that keeps the raster well under GEE's
    ~32 MB limit while not going finer than the sensor's native GSD.

    Target: ~256 pixels per side (plenty for index computation).
    """
    side_m    = buffer_km * 2 * 1000          # AOI side in metres
    raw_scale = int(side_m / 256)             # m/px to hit 256 px/side
    scale     = max(raw_scale, native_res, 30)
    # Round up to nearest 10 m
    scale     = int(np.ceil(scale / 10.0) * 10)
    print(f"[GEE] AOI ~{side_m:.0f}m wide → download scale={scale}m "
          f"(~{side_m / scale:.0f} px/side)")
    return scale


def _build_collection(cfg: dict, aoi, year: int,
                      month_start: int, month_end: int,
                      cloud_max: float):
    """Return a filtered, band-selected ImageCollection."""
    import ee
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


def _download_zip(image, aoi, cfg: dict, scale: int):
    """
    Download a per-band zip via getDownloadURL.
    Returns (C, H, W) float32 array in [0,1] or None.
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

    print(f"[GEE] getDownloadURL at {scale}m …")
    try:
        url = image.getDownloadURL(params)
    except Exception as e:
        print(f"[GEE] getDownloadURL call failed: {e}")
        return None

    print("[GEE] Downloading zip …")
    try:
        r = requests.get(url, timeout=360)
    except Exception as e:
        print(f"[GEE] requests.get failed: {e}")
        return None

    content_type = r.headers.get("Content-Type", "")
    size_kb      = len(r.content) / 1024
    print(f"[GEE] Response: HTTP {r.status_code} | {size_kb:.1f} KB | {content_type}")

    if r.status_code != 200:
        print(f"[GEE] HTTP error: {r.content[:600].decode('utf-8', errors='replace')}")
        return None

    # If GEE returned JSON it's an error message
    if r.content[:1] in (b"{", b"<") or "json" in content_type or "html" in content_type:
        print(f"[GEE] Non-zip response: {r.content[:600].decode('utf-8', errors='replace')}")
        return None

    # Verify zip magic bytes
    if len(r.content) < 4 or r.content[:2] != b"PK":
        print(f"[GEE] Not a zip (no PK header). Preview: {r.content[:200]!r}")
        return None

    # Extract bands
    bands = []
    try:
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            tif_names = sorted(n for n in z.namelist() if n.lower().endswith(".tif"))
            print(f"[GEE] ZIP has {len(tif_names)} TIF(s): {tif_names}")
            if not tif_names:
                print("[GEE] No .tif files in zip")
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
    except zipfile.BadZipFile:
        print(f"[GEE] BadZipFile. Preview: {r.content[:400]!r}")
        return None
    except Exception as e:
        import traceback
        print(f"[GEE] Zip extraction error: {e}")
        traceback.print_exc()
        return None

    if not bands:
        print("[GEE] No bands extracted")
        return None

    array = np.stack(bands, axis=0)      # (C, H, W) — raw DN
    # Apply sensor scaling to get surface reflectance
    array = array * cfg["scale_factor"]
    if "offset" in cfg:
        array = array + cfg["offset"]
    array = np.clip(array, 0.0, 1.0).astype(np.float32)
    print(f"[GEE] Download OK — shape={array.shape}  "
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
    Fetch a cloud-free median composite and return it as a
    (C, H, W) float32 numpy array with values in [0, 1].

    Returns (array, meta) on success, None on failure.
    """
    try:
        import ee
    except ImportError:
        print("[GEE] earthengine-api not installed")
        return None

    sensor = sensor or auto_select_sensor(year)
    cfg    = SENSOR_CONFIGS[sensor]
    print(f"[GEE] sensor={sensor}  year={year}  "
          f"lat={lat:.4f}  lon={lon:.4f}  buffer={buffer_km}km")

    aoi = ee.Geometry.Point([lon, lat]).buffer(buffer_km * 1000).bounds()

    # ── Build collection with progressive cloud relaxation ────
    collection = None
    for cloud_limit in [cfg["cloud_max"], 50, 80]:
        col   = _build_collection(cfg, aoi, year, month_start, month_end, cloud_limit)
        count = col.size().getInfo()
        print(f"[GEE] Scenes (cloud<{cloud_limit}%): {count}")
        if count > 0:
            collection = col
            break

    # Last resort: full year
    if collection is None:
        print("[GEE] Trying full year with cloud<80% …")
        col   = _build_collection(cfg, aoi, year, 1, 12, 80)
        count = col.size().getInfo()
        print(f"[GEE] Full-year scenes: {count}")
        if count == 0:
            print(f"[GEE] No scenes at all for {sensor} {year}")
            return None
        collection = col

    # ── Build median composite (unscaled) ─────────────────────
    image = collection.median().clip(aoi)

    # ── Attempt 1: optimal scale ──────────────────────────────
    scale = _optimal_scale(buffer_km, cfg["resolution"])
    array = _download_zip(image, aoi, cfg, scale)

    # ── Attempt 2: coarser scale ──────────────────────────────
    if array is None:
        coarser = max(scale * 4, 300)
        print(f"[GEE] Retrying at coarser scale {coarser}m …")
        array = _download_zip(image, aoi, cfg, coarser)

    # ── Attempt 3: very coarse fallback ──────────────────────
    if array is None:
        print("[GEE] Retrying at 1000m fallback …")
        array = _download_zip(image, aoi, cfg, 1000)

    if array is None:
        print("[GEE] All download strategies failed")
        return None

    meta = {
        "sensor":   sensor,
        "year":     year,
        "source":   "GEE",
        "n_bands":  array.shape[0],
    }
    return array, meta


# ─────────────────────────────────────────────────────────────
# SAVE HELPER
# ─────────────────────────────────────────────────────────────

def save_array_as_geotiff(array: np.ndarray, meta: dict,
                           output_path: str):
    """Write a (C, H, W) float32 array to a GeoTIFF."""
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
    """Fetch image and save to a temporary GeoTIFF. Returns path or None."""
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
