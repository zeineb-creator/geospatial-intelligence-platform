
"""
gee_connector.py — Google Earth Engine direct image fetcher
============================================================
Fixes applied:
  - save_array_as_geotiff now writes correct CRS (EPSG:4326) and
    affine transform so rasterio can geocode the image and the
    region name / CRS fields display correctly in the UI
  - meta dict carries lat/lon/region_name so app.py can set ic.region
    directly without relying on rasterio geocoding of the saved file
  - computePixels and getDownloadURL both handle GeoTIFF + zip responses
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
# REVERSE GEOCODING
# ─────────────────────────────────────────────────────────────

def _reverse_geocode(lat: float, lon: float) -> str:
    """
    Return a human-readable place name for lat/lon via Nominatim.
    Falls back to coordinate string on any failure.
    """
    try:
        import urllib.request
        ns = "N" if lat >= 0 else "S"
        ew = "E" if lon >= 0 else "W"
        fallback = f"{abs(lat):.3f}°{ns}, {abs(lon):.3f}°{ew}"

        url = (
            f"https://nominatim.openstreetmap.org/reverse"
            f"?lat={lat}&lon={lon}&format=json&zoom=10&accept-language=en"
        )
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "GeoIntelPlatform/3.0 (research)"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())

        addr   = data.get("address", {})
        parts  = []
        for key in ["city", "town", "village", "county", "state", "country"]:
            val = addr.get(key, "")
            if val and sum(1 for c in val if ord(c) < 256) / max(len(val), 1) > 0.6:
                parts.append(val)
                if len(parts) == 2:
                    break
        return ", ".join(parts) if parts else fallback
    except Exception as e:
        print(f"[GEE] Geocoding failed: {e}")
        ns = "N" if lat >= 0 else "S"
        ew = "E" if lon >= 0 else "W"
        return f"{abs(lat):.3f}°{ns}, {abs(lon):.3f}°{ew}"


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
    """Read a GeoTIFF blob into a (C,H,W) float32 array and apply scaling."""
    try:
        from rasterio.io import MemoryFile
        with MemoryFile(raw_bytes) as mf:
            with mf.open() as ds:
                array = ds.read().astype(np.float32)
        print(f"[GEE] read_tiff: shape={array.shape} "
              f"raw=[{array.min():.1f}, {array.max():.1f}]")
        return _apply_scaling(array, cfg)
    except Exception as e:
        print(f"[GEE] _read_tiff_bytes failed: {e}")
        return None


_TIFF_MAGIC = {b'\x49\x49', b'\x4d\x4d'}   # II (LE) or MM (BE)
_ZIP_MAGIC  = b'PK'


# ─────────────────────────────────────────────────────────────
# DOWNLOAD — computePixels  (Strategy 1)
# ─────────────────────────────────────────────────────────────

def _try_compute_pixels(image, aoi, cfg: dict, scale: int):
    try:
        import ee
        print(f"[GEE] computePixels at {scale}m …")

        aoi_info       = aoi.getInfo()
        coords         = aoi_info["coordinates"][0]
        lons           = [c[0] for c in coords]
        lats           = [c[1] for c in coords]
        west,  east    = min(lons), max(lons)
        south, north   = min(lats), max(lats)
        width          = min(1024, max(1, int((east  - west)  * 111320 / scale)))
        height         = min(1024, max(1, int((north - south) * 111320 / scale)))

        params = {
            "expression": image,
            "fileFormat": "GEO_TIFF",
            "bandIds":    cfg["bands"],
            "grid": {
                "crsCode": "EPSG:4326",
                "affineTransform": {
                    "scaleX":     (east  - west)  / width,
                    "scaleY":    -(north - south) / height,
                    "translateX": west,
                    "translateY": north,
                },
                "dimensions": {"width": width, "height": height},
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
# ─────────────────────────────────────────────────────────────

def _try_download_url(image, aoi, cfg: dict, scale: int):
    import requests
    from rasterio.io import MemoryFile

    params = {
        "scale":       scale,
        "region":      aoi.getInfo(),
        "format":      "GEO_TIFF",
        "bands":       cfg["bands"],
        "crs":         "EPSG:4326",
        "filePerBand": False,
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
        return None, f"HTTP {r.status_code} at {scale}m: {body}"

    first = r.content[:1]
    if first in (b"{", b"<") or "json" in ct or "html" in ct:
        body = r.content[:600].decode("utf-8", errors="replace")
        return None, f"GEE error response at {scale}m: {body}"

    magic2 = r.content[:2]

    # ── Direct GeoTIFF ────────────────────────────────────────
    if magic2 in _TIFF_MAGIC:
        print(f"[GEE] Got direct GeoTIFF ({size_kb:.1f} KB)")
        array = _read_tiff_bytes(r.content, cfg)
        if array is None:
            return None, f"Failed to read GeoTIFF at {scale}m"
        if array.shape[0] != len(cfg["bands"]):
            return None, (
                f"Band count mismatch: expected {len(cfg['bands'])}, "
                f"got {array.shape[0]} at {scale}m"
            )
        print(f"[GEE] getDownloadURL(GeoTIFF) OK — shape={array.shape} "
              f"range=[{array.min():.3f}, {array.max():.3f}]")
        return array, None

    # ── Zip of per-band GeoTIFFs ──────────────────────────────
    if magic2 == _ZIP_MAGIC:
        print(f"[GEE] Got zip ({size_kb:.1f} KB)")
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

    preview = r.content[:200].decode("utf-8", errors="replace")
    return None, f"Unknown format at {scale}m (magic={magic2!r}): {preview}"


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
    Returns (array, meta) on success — meta includes lat, lon, region_name,
    crs, and transform so the saved GeoTIFF is properly georeferenced.
    Raises RuntimeError with full diagnostics on failure.
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

    # ── Compute AOI bounding box for transform ────────────────
    aoi_info = aoi.getInfo()
    coords   = aoi_info["coordinates"][0]
    lons_c   = [c[0] for c in coords]
    lats_c   = [c[1] for c in coords]
    west, east   = min(lons_c), max(lons_c)
    south, north = min(lats_c), max(lats_c)

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

    # ── Strategy 2: getDownloadURL ────────────────────────────
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

    # ── Reverse-geocode the actual coordinates ────────────────
    region_name = _reverse_geocode(lat, lon)
    print(f"[GEE] Region: {region_name}")

    # ── Build affine transform for the saved GeoTIFF ──────────
    # rasterio Affine: (pixel_width, 0, west, 0, -pixel_height, north)
    h, w      = array.shape[1], array.shape[2]
    px_w      = (east  - west)  / w
    px_h      = (north - south) / h
    try:
        from rasterio.transform import from_bounds
        transform = from_bounds(west, south, east, north, w, h)
    except Exception:
        transform = None

    meta = {
        "sensor":      sensor,
        "year":        year,
        "source":      "GEE",
        "n_bands":     array.shape[0],
        # Spatial reference — used by save_array_as_geotiff and input_handler
        "lat":         lat,
        "lon":         lon,
        "west":        west,
        "east":        east,
        "south":       south,
        "north":       north,
        "crs":         "EPSG:4326",
        "transform":   transform,
        "width":       w,
        "height":      h,
        "resolution":  (px_w, px_h),
        "region_name": region_name,
    }

    print(f"[GEE] ── done ✓  shape={array.shape}  region={region_name}")
    return array, meta


# ─────────────────────────────────────────────────────────────
# SAVE HELPER — writes CRS + transform into the GeoTIFF
# ─────────────────────────────────────────────────────────────

def save_array_as_geotiff(array: np.ndarray, meta: dict, output_path: str):
    """
    Write (C, H, W) float32 array to a georeferenced GeoTIFF.
    Uses CRS and transform from meta if available so rasterio
    can correctly geocode the image in input_handler.
    """
    try:
        import rasterio
        from rasterio.crs import CRS

        crs_str   = meta.get("crs", "EPSG:4326")
        transform = meta.get("transform", None)

        open_kwargs = dict(
            driver="GTiff",
            height=array.shape[1],
            width=array.shape[2],
            count=array.shape[0],
            dtype=np.float32,
            crs=CRS.from_string(crs_str),
        )
        if transform is not None:
            open_kwargs["transform"] = transform

        with rasterio.open(output_path, "w", **open_kwargs) as dst:
            dst.write(array)

        print(f"[GEE] Saved GeoTIFF: {output_path} "
              f"crs={crs_str} transform={transform}")
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
