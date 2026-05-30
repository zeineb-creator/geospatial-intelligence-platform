"""
gee_connector.py — Google Earth Engine direct image fetcher
============================================================
Returns (array, meta) on success, or raises a descriptive RuntimeError
so app.py can display the exact failure reason in the UI.
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


# ─────────────────────────────────────────────────────────────
# DOWNLOAD — computePixels  (Strategy 1)
# ─────────────────────────────────────────────────────────────

def _try_compute_pixels(image, aoi, cfg: dict, scale: int):
    """
    Uses ee.data.computePixels — returns a raw GeoTIFF blob in memory,
    no zip, no size-limit issues. Requires earthengine-api >= 0.1.374.
    Returns (array, None) on success or (None, error_str) on failure.
    """
    try:
        import ee
        import rasterio
        from rasterio.io import MemoryFile

        print(f"[GEE] computePixels at {scale}m …")

        params = {
            "expression": image.clip(aoi),
            "fileFormat": "GEO_TIFF",
            "bandIds":    cfg["bands"],
            "grid": {
                "crsCode":   "EPSG:4326",
                "scale":     {"xScale": scale / 111320.0,
                              "yScale": scale / 111320.0},
            },
            "region": aoi.getInfo()["coordinates"],
        }

        raw = ee.data.computePixels(params)

        with MemoryFile(raw) as mf:
            with mf.open() as ds:
                array = ds.read().astype(np.float32)

        array = _apply_scaling(array, cfg)
        print(f"[GEE] computePixels OK — shape={array.shape} "
              f"range=[{array.min():.3f}, {array.max():.3f}]")
        return array, None

    except Exception as e:
        msg = f"computePixels({scale}m): {type(e).__name__}: {e}"
        print(f"[GEE] {msg}")
        return None, msg


# ─────────────────────────────────────────────────────────────
# DOWNLOAD — getDownloadURL + zip  (Strategy 2)
# ─────────────────────────────────────────────────────────────

def _try_download_url(image, aoi, cfg: dict, scale: int):
    """
    Uses getDownloadURL — per-band zip download.
    Returns (array, None) on success or (None, error_str) on failure.
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
        msg  = f"HTTP {r.status_code} at {scale}m: {body}"
        print(f"[GEE] {msg}")
        return None, msg

    # GEE error responses arrive as JSON/HTML with HTTP 200
    first = r.content[:1]
    if first in (b"{", b"<") or "json" in ct or "html" in ct:
        body = r.content[:600].decode("utf-8", errors="replace")
        msg  = f"GEE error JSON/HTML at {scale}m: {body}"
        print(f"[GEE] {msg}")
        return None, msg

    if len(r.content) < 4 or r.content[:2] != b"PK":
        preview = r.content[:200].decode("utf-8", errors="replace")
        msg     = f"Not a zip at {scale}m (bad magic bytes). Preview: {preview}"
        print(f"[GEE] {msg}")
        return None, msg

    # Extract bands from zip
    bands = []
    try:
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            tifs = sorted(n for n in z.namelist() if n.lower().endswith(".tif"))
            print(f"[GEE] ZIP contains {len(tifs)} TIF(s): {tifs}")
            if not tifs:
                return None, f"Zip at {scale}m has no .tif files"
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
        preview = r.content[:300].decode("utf-8", errors="replace")
        msg     = f"BadZipFile at {scale}m. Content preview: {preview}"
        print(f"[GEE] {msg}")
        return None, msg
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        msg = f"Zip extraction error at {scale}m: {e}\n{tb}"
        print(f"[GEE] {msg}")
        return None, msg

    if not bands:
        return None, f"No bands extracted from zip at {scale}m"

    array = _apply_scaling(np.stack(bands, axis=0), cfg)
    print(f"[GEE] getDownloadURL OK — shape={array.shape} "
          f"range=[{array.min():.3f}, {array.max():.3f}]")
    return array, None


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
    Raises RuntimeError with a full diagnostic message on failure
    so the caller can display it directly in the UI.
    """
    try:
        import ee
    except ImportError:
        raise RuntimeError("earthengine-api is not installed")

    sensor = sensor or auto_select_sensor(year)
    cfg    = SENSOR_CONFIGS[sensor]
    errors = []   # collect all failure reasons

    print(f"[GEE] ── fetch ─ sensor={sensor} year={year} "
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
        # Full-year sweep
        col   = _build_collection(cfg, aoi, year, 1, 12, 80)
        count = col.size().getInfo()
        print(f"[GEE] Full-year scenes cloud<80%: {count}")
        if count == 0:
            # Sentinel→Landsat automatic fallback
            if "Sentinel" in sensor:
                fb = "Landsat 8/9" if year >= 2013 else "Landsat 5 TM"
                print(f"[GEE] S2 empty — retrying with {fb}")
                return fetch_image_as_array(
                    lat=lat, lon=lon, year=year,
                    month_start=month_start, month_end=month_end,
                    sensor=fb, buffer_km=buffer_km,
                )
            raise RuntimeError(
                f"No satellite scenes found for sensor={sensor}, year={year}, "
                f"lat={lat:.4f}, lon={lon:.4f}. "
                "Try a different year, a larger buffer, or a different sensor."
            )
        collection = col

    image = collection.median().clip(aoi)

    # ── Strategy 1: computePixels ─────────────────────────────
    base_scale = _scale_for_aoi(buffer_km, cfg["resolution"])
    array, err = _try_compute_pixels(image, aoi, cfg, base_scale)
    if err:
        errors.append(err)

    # ── Strategy 2: getDownloadURL at multiple scales ─────────
    if array is None:
        for scale in sorted(set([base_scale, max(base_scale * 4, 300), 1000])):
            array, err = _try_download_url(image, aoi, cfg, scale)
            if err:
                errors.append(err)
            if array is not None:
                break

    if array is None:
        diagnosis = "\n".join(f"  • {e}" for e in errors)
        raise RuntimeError(
            f"All GEE download strategies failed for sensor={sensor}, year={year}.\n"
            f"Errors:\n{diagnosis}\n\n"
            "Common causes:\n"
            "  1. Image payload too large — reduce 'Area radius' in the sidebar\n"
            "  2. GEE compute quota exceeded — wait a few minutes and retry\n"
            "  3. Collection has no SR data for this tile (check GEE Explorer)\n"
            "  4. Service account lacks 'Earth Engine Resource Viewer' role"
        )

    meta = {"sensor": sensor, "year": year, "source": "GEE", "n_bands": array.shape[0]}
    print(f"[GEE] ── done ✓ shape={array.shape}")
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
    """Fetch and save to a temp GeoTIFF. Returns path or None."""
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
