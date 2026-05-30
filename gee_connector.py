"""
gee_connector.py — Google Earth Engine direct image fetcher
============================================================
Fixed based on observed errors:
  1. computePixels: fixed grid params format for current earthengine-api
  2. getDownloadURL: MM* magic bytes = GeoTIFF (not zip) — read directly
     GEE sometimes returns a single multi-band GeoTIFF instead of a zip
     regardless of filePerBand setting. Both cases are now handled.
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
    try:
        import ee
    except ImportError:
        print("[GEE] earthengine-api not installed")
        return False

    try:
        import streamlit as st
        sa_info = st.secrets.get("GEE_SERVICE_ACCOUNT", None)
        if sa_info:
            private_key = sa_info.get("private_key", "").replace("\\n", "\n")
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
            if ee.Number(42).getInfo() == 42:
                print("[GEE] Initialised via service account ✓")
                return True
            return False
    except Exception as e:
        print(f"[GEE] Service account init failed: {e}")

    try:
        import ee
        ee.Initialize()
        if ee.Number(1).getInfo() == 1:
            print("[GEE] Initialised via default credentials ✓")
            return True
    except Exception as e:
        print(f"[GEE] Default credentials failed: {e}")

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
# INTERNAL HELPERS
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
    return np.clip(
        array.astype(np.float32) * cfg["scale_factor"] + cfg.get("offset", 0.0),
        0.0, 1.0,
    )


def _scale_for_aoi(buffer_km: float, native_res: int, target_px: int = 256) -> int:
    side_m = buffer_km * 2 * 1000
    scale  = max(int(side_m / target_px), native_res, 30)
    return int(np.ceil(scale / 10.0) * 10)


def _read_tiff_bytes(raw_bytes: bytes, cfg: dict):
    """
    Read a GeoTIFF from raw bytes (in-memory).
    Returns (C,H,W) float32 array after scaling, or None.
    Handles both single-band-per-file and multi-band GeoTIFFs.
    """
    try:
        import rasterio
        from rasterio.io import MemoryFile
        with MemoryFile(raw_bytes) as mf:
            with mf.open() as ds:
                array = ds.read().astype(np.float32)  # shape: (bands, H, W)
        print(f"[GEE] read_tiff_bytes: shape={array.shape} "
              f"dtype=float32 raw_range=[{array.min():.1f}, {array.max():.1f}]")
        return _apply_scaling(array, cfg)
    except Exception as e:
        print(f"[GEE] _read_tiff_bytes failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# DOWNLOAD — computePixels  (Strategy 1)
# Fixed: use correct params format for current earthengine-api
# ─────────────────────────────────────────────────────────────

def _try_compute_pixels(image, aoi, cfg: dict, scale: int):
    """
    Uses ee.data.computePixels with corrected parameter format.
    Returns (array, None) on success or (None, error_str) on failure.
    """
    try:
        import ee
        print(f"[GEE] computePixels at {scale}m …")

        # Correct format: use 'dimensions' not 'scale' inside grid
        aoi_info   = aoi.getInfo()
        coords     = aoi_info["coordinates"][0]
        lons       = [c[0] for c in coords]
        lats       = [c[1] for c in coords]
        west, east = min(lons), max(lons)
        south, north = min(lats), max(lats)

        width  = max(1, int((east  - west)  * 111320 / scale))
        height = max(1, int((north - south) * 111320 / scale))
        # Cap at 1024 px per side to stay within quota
        width  = min(width,  1024)
        height = min(height, 1024)

        params = {
            "expression": image,
            "fileFormat": "GEO_TIFF",
            "bandIds":    cfg["bands"],
            "grid": {
                "crsCode":    "EPSG:4326",
                "affineTransform": {
                    "scaleX":      (east - west)  / width,
                    "scaleY":     -(north - south) / height,
                    "translateX":  west,
                    "translateY":  north,
                },
                "dimensions": {
                    "width":  width,
                    "height": height,
                },
            },
        }

        raw   = ee.data.computePixels(params)
        array = _read_tiff_bytes(raw, cfg)
        if array is None:
            return None, f"computePixels({scale}m): _read_tiff_bytes returned None"

        print(f"[GEE] computePixels OK — shape={array.shape} "
              f"range=[{array.min():.3f}, {array.max():.3f}]")
        return array, None

    except Exception as e:
        msg = f"computePixels({scale}m): {type(e).__name__}: {e}"
        print(f"[GEE] {msg}")
        return None, msg


# ─────────────────────────────────────────────────────────────
# DOWNLOAD — getDownloadURL  (Strategy 2)
# Fixed: MM* magic bytes = GeoTIFF, not zip — read directly
# ─────────────────────────────────────────────────────────────

# TIFF magic bytes: II (little-endian) = b'\x49\x49' or MM (big-endian) = b'\x4d\x4d'
_TIFF_MAGIC = {b'\x49\x49', b'\x4d\x4d'}
_ZIP_MAGIC  = b'PK'


def _try_download_url(image, aoi, cfg: dict, scale: int):
    """
    Uses getDownloadURL.
    GEE may return either a zip-of-TIFFs OR a single multi-band GeoTIFF.
    Both cases are handled.
    Returns (array, None) on success or (None, error_str) on failure.
    """
    import requests
    import rasterio
    from rasterio.io import MemoryFile

    # Request multi-band GeoTIFF directly — more reliable than per-band zip
    params = {
        "scale":       scale,
        "region":      aoi.getInfo(),
        "format":      "GEO_TIFF",
        "bands":       cfg["bands"],
        "crs":         "EPSG:4326",
        "filePerBand": False,          # ← request single multi-band GeoTIFF
    }

    print(f"[GEE] getDownloadURL at {scale}m (multi-band GeoTIFF) …")
    try:
        url = image.getDownloadURL(params)
    except Exception as e:
        msg = f"getDownloadURL RPC({scale}m): {type(e).__name__}: {e}"
        print(f"[GEE] {msg}")
        return None, msg

    try:
        r = requests.get(url, timeout=360)
    except Exception as e:
        msg = f"HTTP GET({scale}m): {type(e).__name__}: {e}"
        print(f"[GEE] {msg}")
        return None, msg

    ct      = r.headers.get("Content-Type", "")
    size_kb = len(r.content) / 1024
    print(f"[GEE] HTTP {r.status_code} | {size_kb:.1f} KB | {ct}")

    if r.status_code != 200:
        body = r.content[:600].decode("utf-8", errors="replace")
        return None, f"HTTP {r.status_code} at {scale}m: {body}"

    # GEE error responses come as JSON/HTML
    first = r.content[:1]
    if first in (b"{", b"<") or "json" in ct or "html" in ct:
        body = r.content[:600].decode("utf-8", errors="replace")
        return None, f"GEE error response at {scale}m: {body}"

    magic2 = r.content[:2]

    # ── Case A: GeoTIFF returned directly ────────────────────
    if magic2 in _TIFF_MAGIC:
        print(f"[GEE] Got direct GeoTIFF ({size_kb:.1f} KB)")
        array = _read_tiff_bytes(r.content, cfg)
        if array is None:
            return None, f"Failed to read direct GeoTIFF at {scale}m"
        if array.shape[0] != len(cfg["bands"]):
            return None, (
                f"Band count mismatch: expected {len(cfg['bands'])}, "
                f"got {array.shape[0]} at {scale}m"
            )
        print(f"[GEE] getDownloadURL(GeoTIFF) OK — shape={array.shape} "
              f"range=[{array.min():.3f}, {array.max():.3f}]")
        return array, None

    # ── Case B: zip of per-band GeoTIFFs ─────────────────────
    if magic2 == _ZIP_MAGIC:
        print(f"[GEE] Got zip ({size_kb:.1f} KB), extracting bands …")
        bands = []
        try:
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                tifs = sorted(n for n in z.namelist() if n.lower().endswith(".tif"))
                print(f"[GEE] ZIP: {len(tifs)} TIF(s): {tifs}")
                if not tifs:
                    return None, f"Zip at {scale}m has no .tif files"
                for name in tifs:
                    data = z.read(name)
                    with MemoryFile(data) as mf:
                        with mf.open() as ds:
                            bands.append(ds.read(1).astype(np.float32))
        except zipfile.BadZipFile:
            preview = r.content[:300].decode("utf-8", errors="replace")
            return None, f"BadZipFile at {scale}m: {preview}"
        except Exception as e:
            return None, f"Zip extraction error at {scale}m: {e}"

        if not bands:
            return None, f"No bands from zip at {scale}m"

        array = _apply_scaling(np.stack(bands, axis=0), cfg)
        print(f"[GEE] getDownloadURL(zip) OK — shape={array.shape} "
              f"range=[{array.min():.3f}, {array.max():.3f}]")
        return array, None

    # ── Unknown format ────────────────────────────────────────
    preview = r.content[:200].decode("utf-8", errors="replace")
    return None, f"Unknown response format at {scale}m (magic={magic2!r}): {preview}"


# ─────────────────────────────────────────────────────────────
# MAIN FETCH
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
    Fetch a cloud-free median composite.
    Returns (array, meta) on success.
    Raises RuntimeError with full diagnostics on failure.
    array is (C, H, W) float32 in [0, 1].
    """
    try:
        import ee
    except ImportError:
        raise RuntimeError("earthengine-api is not installed")

    sensor = sensor or auto_select_sensor(year)
    cfg    = SENSOR_CONFIGS[sensor]
    errors = []

    print(f"[GEE] ── fetch: sensor={sensor} year={year} "
          f"lat={lat:.4f} lon={lon:.4f} buffer={buffer_km}km")

    aoi = ee.Geometry.Point([lon, lat]).buffer(buffer_km * 1000).bounds()

    # ── Find a non-empty collection ───────────────────────────
    collection = None
    for cloud_limit in [cfg["cloud_max"], 50, 80]:
        col   = _build_collection(cfg, aoi, year, month_start, month_end, cloud_limit)
        count = col.size().getInfo()
        print(f"[GEE] Scenes cloud<{cloud_limit}%: {count}")
        if count > 0:
            collection = col
            break

    if collection is None:
        col   = _build_collection(cfg, aoi, year, 1, 12, 80)
        count = col.size().getInfo()
        print(f"[GEE] Full-year scenes: {count}")
        if count == 0:
            if "Sentinel" in sensor:
                fb = "Landsat 8/9" if year >= 2013 else "Landsat 5 TM"
                print(f"[GEE] S2 empty → falling back to {fb}")
                return fetch_image_as_array(
                    lat=lat, lon=lon, year=year,
                    month_start=month_start, month_end=month_end,
                    sensor=fb, buffer_km=buffer_km,
                )
            raise RuntimeError(
                f"No scenes found: sensor={sensor}, year={year}, "
                f"lat={lat:.4f}, lon={lon:.4f}. "
                "Try a different year, larger buffer, or different sensor."
            )
        collection = col

    image = collection.median().clip(aoi)

    base_scale = _scale_for_aoi(buffer_km, cfg["resolution"])
    scales     = sorted(set([base_scale, max(base_scale * 4, 300), 1000]))

    # ── Strategy 1: computePixels ─────────────────────────────
    array, err = _try_compute_pixels(image, aoi, cfg, base_scale)
    if err:
        errors.append(err)

    # ── Strategy 2: getDownloadURL at multiple scales ─────────
    if array is None:
        for scale in scales:
            array, err = _try_download_url(image, aoi, cfg, scale)
            if err:
                errors.append(err)
            if array is not None:
                break

    if array is None:
        diagnosis = "\n".join(f"  • {e}" for e in errors)
        raise RuntimeError(
            f"All download strategies failed for sensor={sensor}, year={year}.\n"
            f"Errors:\n{diagnosis}\n\n"
            "Suggestions:\n"
            "  1. Reduce 'Area radius' to 2–3 km in the sidebar\n"
            "  2. Try Landsat 8/9 instead of Sentinel-2 for pre-2017 years\n"
            "  3. GEE compute quota may be exhausted — wait a few minutes\n"
            "  4. Verify service account has 'Earth Engine Resource Viewer' role"
        )

    meta = {"sensor": sensor, "year": year, "source": "GEE", "n_bands": array.shape[0]}
    print(f"[GEE] ── done ✓  shape={array.shape}")
    return array, meta


# ─────────────────────────────────────────────────────────────
# SAVE HELPER
# ─────────────────────────────────────────────────────────────

def save_array_as_geotiff(array: np.ndarray, meta: dict, output_path: str):
    try:
        import rasterio
        with rasterio.open(
            output_path, "w", driver="GTiff",
            height=array.shape[1], width=array.shape[2],
            count=array.shape[0], dtype=np.float32,
        ) as dst:
            dst.write(array)
        return output_path
    except Exception as e:
        print(f"[GEE] save_array_as_geotiff failed: {e}")
        return None


def fetch_and_save(lat, lon, year, month_start=1, month_end=12,
                   sensor=None, buffer_km=5.0):
    try:
        result = fetch_image_as_array(
            lat=lat, lon=lon, year=year,
            month_start=month_start, month_end=month_end,
            sensor=sensor, buffer_km=buffer_km,
        )
    except RuntimeError as e:
        print(f"[GEE] fetch_and_save error: {e}")
        return None
    if result is None:
        return None
    array, meta = result
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        path = tmp.name
    return save_array_as_geotiff(array, meta, path)
