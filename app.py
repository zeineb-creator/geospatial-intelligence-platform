Here is the complete `app.py` with the GEE debug section added:

```python
"""
app.py — Multimodal Geospatial Intelligence Platform
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io, sys, os, tempfile, re
from datetime import date as _date

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Geospatial Intelligence Platform",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — light theme ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.stApp { background-color: #f5f7fa; color: #1a1f2e; }

section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #dde3ed;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div.stMarkdown { color: #1a1f2e !important; }

h1, h2, h3 { font-family: 'IBM Plex Mono', monospace !important; letter-spacing: -0.02em; }
h1 { font-size: 1.6rem !important; color: #1a1f2e !important; }
h2 { font-size: 1.05rem !important; color: #2563eb !important; border-bottom: 1px solid #dde3ed; padding-bottom: 0.4rem; }
h3 { font-size: 0.95rem !important; color: #3b82f6 !important; }

.geo-card { background: #ffffff; border: 1px solid #dde3ed; border-radius: 8px; padding: 1.1rem 1.3rem; margin-bottom: 1rem; }
.geo-card-accent { border-left: 3px solid #2563eb; }
.geo-card-warn   { border-left: 3px solid #d97706; background: #fffbeb; }
.geo-card-good   { border-left: 3px solid #16a34a; background: #f0fdf4; }
.geo-card-error  { border-left: 3px solid #dc2626; background: #fef2f2; }

.metric-row { display: flex; gap: 0.8rem; flex-wrap: wrap; margin: 0.6rem 0; }
.metric-chip { background: #f0f4ff; border-radius: 6px; padding: 0.5rem 0.9rem; font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; color: #2563eb; border: 1px solid #c7d7fc; }
.metric-chip span { color: #1a1f2e; font-weight: 600; margin-left: 0.4rem; }

.conf-bar-container { background: #e5e7eb; border-radius: 999px; height: 8px; width: 100%; overflow: hidden; margin-top: 0.4rem; }
.conf-bar-fill { height: 100%; border-radius: 999px; transition: width 0.8s ease; }

.anomaly-tag { display: inline-block; background: #fffbeb; border: 1px solid #d97706; border-radius: 4px; padding: 0.25rem 0.6rem; font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: #92400e; margin: 0.2rem 0.2rem 0.2rem 0; }

.report-section { background: #ffffff; border: 1px solid #dde3ed; border-radius: 6px; padding: 1rem 1.3rem; margin-bottom: 0.8rem; font-size: 0.88rem; line-height: 1.75; color: #374151; }
.report-section h4 { font-family: 'IBM Plex Mono', monospace; color: #2563eb; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; margin: 0 0 0.6rem 0; }

.lc-bar { display: flex; border-radius: 4px; overflow: hidden; height: 14px; width: 100%; }
.lc-water  { background: #3b82f6; }
.lc-veg    { background: #22c55e; }
.lc-urban  { background: #f59e0b; }
.lc-barren { background: #94a3b8; }

.label-mono { font-family: 'IBM Plex Mono', monospace; font-size: 0.70rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.1em; }
.geo-divider { border: none; border-top: 1px solid #dde3ed; margin: 1.2rem 0; }

[data-testid="stFileUploader"] { background: #f8fafc !important; border: 1px dashed #c7d7fc !important; border-radius: 8px !important; }

.stButton > button { background: #2563eb !important; color: #fff !important; border: none !important; border-radius: 6px !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.85rem !important; padding: 0.5rem 1.5rem !important; transition: background 0.2s; }
.stButton > button:hover { background: #1d4ed8 !important; }

.stTabs [data-baseweb="tab"] { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: #6b7280; }
.stTabs [aria-selected="true"] { color: #2563eb !important; border-bottom-color: #2563eb !important; }

.streamlit-expanderHeader { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.82rem !important; color: #2563eb !important; background: #f8fafc !important; }

.meta-row { display: flex; justify-content: space-between; padding: 0.35rem 0; border-bottom: 1px solid #f0f4ff; font-size: 0.82rem; }

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

[data-testid="collapsedControl"] { visibility: visible !important; display: flex !important; }
button[kind="header"] { visibility: visible !important; }
[data-testid="stSidebarCollapseButton"] { visibility: visible !important; }
</style>
""", unsafe_allow_html=True)

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))


# ── Utility helpers ───────────────────────────────────────────────────────────

def save_upload_to_temp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def _extract_lat_from_meta(meta: dict):
    try:
        import rasterio.transform as rt
        t = meta.get("transform")
        h = meta.get("height", 100)
        w = meta.get("width", 100)
        if t:
            _, y = rt.xy(t, h // 2, w // 2)
            return float(y)
    except Exception:
        pass
    return None


def _extract_lon_from_meta(meta: dict):
    try:
        import rasterio.transform as rt
        t = meta.get("transform")
        h = meta.get("height", 100)
        w = meta.get("width", 100)
        if t:
            x, _ = rt.xy(t, h // 2, w // 2)
            return float(x)
    except Exception:
        pass
    return None


def render_index_map(arr, title, cmap):
    fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="#ffffff")
    ax.set_facecolor("#f8fafc")
    im = ax.imshow(arr, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(colors="#6b7280", labelsize=7)
    cbar.outline.set_edgecolor("#dde3ed")
    ax.set_title(title, color="#2563eb", fontsize=9, fontfamily="monospace", pad=6)
    ax.axis("off")
    fig.tight_layout(pad=0.5)
    return fig


def render_land_cover_bar(land_cover: dict) -> str:
    parts = ""
    for key, pct in land_cover.items():
        if pct > 0:
            parts += f'<div class="lc-{key}" style="width:{pct:.1f}%"></div>'
    return f'<div class="lc-bar">{parts}</div>'


def render_confidence_bar(score: float) -> str:
    color = "#16a34a" if score >= 75 else "#d97706" if score >= 55 else "#dc2626"
    return f'<div class="conf-bar-container"><div class="conf-bar-fill" style="width:{score:.0f}%;background:{color};"></div></div>'


def parse_report_sections(report_text: str) -> dict:
    sections = {}
    parts = re.split(r'\n##\s+', report_text)
    for part in parts:
        if not part.strip():
            continue
        lines = part.strip().split("\n", 1)
        title = lines[0].strip().lstrip("#").strip()
        content = lines[1].strip() if len(lines) > 1 else ""
        sections[title] = content
    return sections


def render_report_section(title: str, content: str, icon: str = ""):
    st.markdown(f"""
    <div class="report-section">
        <h4>{icon} {title}</h4>
        {content.replace(chr(10), '<br>')}
    </div>
    """, unsafe_allow_html=True)


SECTION_ICONS = {
    "Executive": "📋", "Vegetation": "🌿", "Temporal": "📈",
    "Hydrological": "💧", "Climate": "🌡️", "Aridity": "☀️",
    "Key Findings": "🔍", "Monitoring": "📡", "Confidence": "⚖️",
}


def get_icon(title):
    for key, icon in SECTION_ICONS.items():
        if key.lower() in title.lower():
            return icon
    return "📄"


def import_error_card(module: str, err: Exception):
    st.markdown(f"""
    <div class="geo-card geo-card-error">
        <strong>Import failed:</strong> <code>{module}</code><br><br>
        <code style="font-size:0.78rem;">{err}</code><br><br>
        <span style="color:#6b7280;font-size:0.78rem;">
            Check that the function name in <code>{module}</code> matches what app.py expects.
        </span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="padding:0.5rem 0 1.2rem 0;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:1.05rem;color:#2563eb;font-weight:600;">🛰️ GeoIntel</div>
        <div style="font-size:0.70rem;color:#6b7280;margin-top:0.2rem;letter-spacing:0.05em;">MULTIMODAL GEOSPATIAL PLATFORM</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📡 Image Input")
    input_mode = st.radio(
        "Source",
        ["Upload image(s)", "Fetch from GEE"],
        horizontal=True,
        help="Upload your own GeoTIFF, or fetch directly from Google Earth Engine."
    )

    uploaded_t1 = None
    uploaded_t2 = None
    gee_params  = None

    if input_mode == "Upload image(s)":
        uploaded_t1 = st.file_uploader(
            "Primary image (or earlier date)",
            type=["tif", "tiff", "png", "jpg"], key="img_t1",
            help="GeoTIFF preferred — 6-band Sentinel-2 or Landsat recommended."
        )
        uploaded_t2 = st.file_uploader(
            "Second image — optional (later date)",
            type=["tif", "tiff", "png", "jpg"], key="img_t2",
            help="Upload for temporal NDVI change detection."
        )
        if uploaded_t1 is not None or uploaded_t2 is not None:
            st.markdown("#### 📅 Acquisition Dates")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                date_t1 = st.date_input("Image 1", value=None, key="date_t1")
            with col_d2:
                date_t2 = st.date_input("Image 2", value=None, key="date_t2")
        else:
            date_t1 = date_t2 = None

    else:  # GEE mode
        st.markdown("**Location**")
        col_lat, col_lon = st.columns(2)
        with col_lat:
            gee_lat = st.number_input("Latitude",  value=36.45, format="%.4f", key="gee_lat")
        with col_lon:
            gee_lon = st.number_input("Longitude", value=10.73, format="%.4f", key="gee_lon")

        st.markdown("**Temporal range**")
        col_y1, col_y2 = st.columns(2)
        with col_y1:
            gee_year1 = st.number_input("Year 1 (earlier)", value=2015,
                                        min_value=1984, max_value=2024, key="gee_y1")
        with col_y2:
            gee_year2 = st.number_input("Year 2 (later)",   value=2023,
                                        min_value=1984, max_value=2024, key="gee_y2")

        gee_sensor = st.selectbox(
            "Sensor (auto if blank)",
            ["Auto", "Sentinel-2 L2A", "Landsat 8/9", "Landsat 5 TM"],
            key="gee_sensor"
        )
        gee_buffer = st.slider("Area radius (km)", 2.0, 20.0, 5.0, 0.5, key="gee_buf")

        gee_params = {
            "lat":       gee_lat,
            "lon":       gee_lon,
            "year1":     int(gee_year1),
            "year2":     int(gee_year2),
            "sensor":    None if gee_sensor == "Auto" else gee_sensor,
            "buffer_km": gee_buffer,
        }
        date_t1 = _date(int(gee_year1), 4, 1)
        date_t2 = _date(int(gee_year2), 4, 1)

    st.markdown("<hr class='geo-divider'>", unsafe_allow_html=True)
    st.markdown("### 📊 Climate Data")
    climate_mode = st.radio(
        "Source",
        ["Auto-fetch (NASA POWER)", "Upload CSV"],
        horizontal=True,
    )
    uploaded_csv = None
    if climate_mode == "Upload CSV":
        uploaded_csv = st.file_uploader("NASA POWER CSV", type=["csv"], key="csv")

    fetch_timeseries = st.checkbox(
        "📈 Fetch NDVI time series (GEE required)",
        value=False,
    )

    st.markdown("<hr class='geo-divider'>", unsafe_allow_html=True)
    st.markdown("### ⚙️ Options")
    user_question = st.text_area(
        "Custom question (optional)",
        value="Provide a full scientific interpretation of this image and data.",
        height=80
    )
    run_btn = st.button("▶ Run Analysis", use_container_width=True)

    st.markdown("<hr class='geo-divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.68rem;color:#9ca3af;line-height:1.7;">
        Pipeline: Image → ViT → RAG → LLM<br>
        LLM: Groq / llama-3.3-70b-versatile<br>
        Indices: NDVI · NDWI · NDBI · BSI · UI · MNDWI
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="margin-bottom:1.5rem;">
    <h1 style="margin:0;">Geospatial Intelligence Platform</h1>
    <div class="label-mono" style="margin-top:0.3rem;">Satellite imagery · Spectral analysis · AI-generated environmental reports</div>
</div>
""", unsafe_allow_html=True)

if not run_btn:
    st.markdown("""
    <div class="geo-card" style="text-align:center;padding:3rem 2rem;border-style:dashed;">
        <div style="font-size:2.5rem;margin-bottom:1rem;">🛰️</div>
        <div style="font-family:'IBM Plex Mono',monospace;color:#2563eb;font-size:1rem;margin-bottom:0.5rem;">Ready for Analysis</div>
        <div style="color:#6b7280;font-size:0.85rem;max-width:420px;margin:0 auto;">
            Upload a satellite image in the sidebar, optionally add a second image for temporal
            comparison and a NASA POWER CSV, then click <strong>Run Analysis</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if not uploaded_t1 and not gee_params:
    st.markdown("""
    <div class="geo-card geo-card-error">
        ⚠️ <strong>No image source.</strong> Upload an image or configure GEE fetch in the sidebar.
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

pipeline_warnings = []
status = st.status("Running geospatial analysis pipeline…", expanded=True)
temp_files = []

try:
    with status:

        # ── Lazy imports ──────────────────────────────────────────────────────
        st.write("📦 Loading pipeline modules…")

        try:
            from geospatial_platform.context import InputContext
        except Exception as e:
            import_error_card("geospatial_platform.context → InputContext", e); st.stop()

        try:
            from geospatial_platform.input_handler import build_input_context
        except Exception as e:
            import_error_card("geospatial_platform.input_handler → build_input_context", e); st.stop()

        try:
            from geospatial_platform.input_handler import load_image
        except Exception as e:
            import_error_card("geospatial_platform.input_handler → load_image", e); st.stop()

        try:
            from geospatial_platform.image_processor import process_image
        except Exception as e:
            import_error_card("geospatial_platform.image_processor → process_image", e); st.stop()

        try:
            from geospatial_platform.vision_model import extract_vit_features
        except Exception as e:
            import_error_card("geospatial_platform.vision_model → extract_vit_features", e); st.stop()

        build_climate_summary = None
        populate_convenience_fields = None
        try:
            from geospatial_platform.data_integrator import (
                integrate_data, build_climate_summary, populate_convenience_fields,
            )
        except ImportError:
            try:
                from geospatial_platform.data_integrator import integrate_data
                st.warning("⚠️ build_climate_summary / populate_convenience_fields not found.")
            except Exception as e:
                import_error_card("geospatial_platform.data_integrator → integrate_data", e); st.stop()

        try:
            from geospatial_platform.rag import retrieve_context
        except Exception as e:
            import_error_card("geospatial_platform.rag → retrieve_context", e); st.stop()

        try:
            from geospatial_platform.llm_engine import generate_report
        except Exception as e:
            import_error_card("geospatial_platform.llm_engine → generate_report", e); st.stop()

        # ── GEE fetch ─────────────────────────────────────────────────────────
        path_t1 = path_t2 = path_csv = None

        if gee_params:
            st.write("🛰️ Fetching imagery from Google Earth Engine…")
            try:
                from gee_connector import (
                    init_gee, fetch_image_as_array, fetch_and_save,
                    auto_select_sensor, SENSOR_CONFIGS
                )

                # ── DEBUG: surface GEE secrets in UI ──────────────────────────
                st.write("🔍 Checking GEE secrets…")
                try:
                    sa_info = st.secrets.get("GEE_SERVICE_ACCOUNT", None)
                    if sa_info is None:
                        st.error("❌ GEE_SERVICE_ACCOUNT section not found in Streamlit secrets.")
                        st.stop()
                    else:
                        keys_present = list(sa_info.keys())
                        st.write(f"  Secret keys found: `{keys_present}`")
                        st.write(f"  client_email: `{sa_info.get('client_email', 'MISSING')}`")
                        st.write(f"  project_id:   `{sa_info.get('project_id', 'MISSING')}`")
                        st.write(f"  private_key_id: `{sa_info.get('private_key_id', 'MISSING')}`")
                        raw_key = sa_info.get("private_key", "")
                        st.write(f"  private_key length: `{len(raw_key)}` chars")
                        st.write(f"  private_key starts with: `{raw_key[:50]!r}`")
                        st.write(f"  private_key ends with:   `{raw_key[-30:]!r}`")
                        has_header = "BEGIN PRIVATE KEY" in raw_key
                        has_literal_n = "\\n" in raw_key
                        has_real_n = "\n" in raw_key
                        st.write(f"  Has 'BEGIN PRIVATE KEY': `{has_header}`")
                        st.write(f"  Has literal \\\\n: `{has_literal_n}`")
                        st.write(f"  Has real newline: `{has_real_n}`")
                        if not has_header:
                            st.error("❌ private_key is malformed — missing 'BEGIN PRIVATE KEY'")
                            st.stop()
                except Exception as debug_e:
                    st.error(f"❌ Secret inspection failed: {debug_e}")
                    st.stop()

                # ── Attempt GEE init ──────────────────────────────────────────
                st.write("🔐 Initialising GEE…")
                gee_ok = False
                try:
                    gee_ok = init_gee()
                except Exception as init_e:
                    import traceback
                    st.error(f"❌ init_gee() raised an exception: {init_e}")
                    st.code(traceback.format_exc())
                    st.stop()

                if not gee_ok:
                    st.error(
                        "❌ GEE initialisation failed. "
                        "Check Streamlit Cloud logs (Manage app → Logs) "
                        "for [GEE] print statements showing the exact error."
                    )
                    st.info(
                        "Common causes:\n"
                        "1. `private_key` has wrong newline format (needs `\\n` not real newlines)\n"
                        "2. `private_key_id` is missing or wrong\n"
                        "3. Earth Engine API not enabled in Google Cloud project\n"
                        "4. Service account doesn't have Earth Engine access\n"
                        "5. New key not yet propagated (wait 1–2 min and retry)"
                    )
                    st.stop()

                st.write("✅ GEE initialized")

                sensor1 = gee_params.get("sensor") or auto_select_sensor(gee_params["year1"])
                st.write(
                    f"Sensor: `{sensor1}` | Year: `{gee_params['year1']}` | "
                    f"Lat: `{gee_params['lat']:.4f}` | Lon: `{gee_params['lon']:.4f}`"
                )

                # ── Scene count check ─────────────────────────────────────────
                try:
                    import ee
                    point = ee.Geometry.Point([gee_params["lon"], gee_params["lat"]])
                    aoi   = point.buffer(gee_params["buffer_km"] * 1000).bounds()
                    cfg   = SENSOR_CONFIGS[sensor1]
                    col   = (
                        ee.ImageCollection(cfg["collection"])
                        .filterBounds(aoi)
                        .filterDate(
                            f"{gee_params['year1']}-01-01",
                            f"{gee_params['year1']}-12-31",
                        )
                        .filter(ee.Filter.lt(cfg["cloud_prop"], 80))
                    )
                    count = col.size().getInfo()
                    st.write(f"📡 Scenes available for Year 1 (cloud < 80%): `{count}`")
                    if count == 0:
                        st.error(
                            "❌ No satellite scenes found. "
                            "Try a different year or larger buffer radius."
                        )
                        st.stop()
                except Exception as scene_e:
                    st.warning(f"Scene count check failed (non-fatal): {scene_e}")

                # ── Fetch image 1 ─────────────────────────────────────────────
                st.write(f"  Downloading image data for `{gee_params['year1']}`…")
                result1 = fetch_image_as_array(
                    lat=gee_params["lat"],
                    lon=gee_params["lon"],
                    year=gee_params["year1"],
                    sensor=sensor1,
                    buffer_km=gee_params["buffer_km"],
                )

                if result1 is None:
                    st.error(
                        "❌ GEE image download failed. "
                        "Check Streamlit Cloud logs for [GEE] traceback."
                    )
                    st.stop()

                array_t1, meta_t1 = result1
                with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
                    path_t1 = tmp.name
                from gee_connector import save_array_as_geotiff
                path_t1 = save_array_as_geotiff(array_t1, meta_t1, path_t1)
                temp_files.append(path_t1)
                st.write(f"  ✓ Image 1 (`{gee_params['year1']}`) downloaded — shape: `{array_t1.shape}`")

                # ── Fetch image 2 ─────────────────────────────────────────────
                if gee_params["year2"] != gee_params["year1"]:
                    sensor2 = gee_params.get("sensor") or auto_select_sensor(gee_params["year2"])
                    st.write(f"  Downloading image data for `{gee_params['year2']}`…")
                    result2 = fetch_image_as_array(
                        lat=gee_params["lat"],
                        lon=gee_params["lon"],
                        year=gee_params["year2"],
                        sensor=sensor2,
                        buffer_km=gee_params["buffer_km"],
                    )
                    if result2 is not None:
                        array_t2, meta_t2 = result2
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
                            path_t2 = tmp.name
                        path_t2 = save_array_as_geotiff(array_t2, meta_t2, path_t2)
                        temp_files.append(path_t2)
                        st.write(f"  ✓ Image 2 (`{gee_params['year2']}`) downloaded — shape: `{array_t2.shape}`")
                    else:
                        pipeline_warnings.append(
                            f"GEE Image 2 ({gee_params['year2']}) fetch failed — temporal analysis skipped."
                        )

            except Exception as e:
                import traceback
                st.error(f"❌ GEE error: {e}")
                st.code(traceback.format_exc())
                st.stop()

        else:
            st.write("📥 Saving uploaded files…")
            if uploaded_t1:
                path_t1 = save_upload_to_temp(uploaded_t1)
                temp_files.append(path_t1)
            if uploaded_t2:
                path_t2 = save_upload_to_temp(uploaded_t2)
                temp_files.append(path_t2)

        if uploaded_csv:
            path_csv = save_upload_to_temp(uploaded_csv)
            temp_files.append(path_csv)

        # ── Step 1 — Input handling ───────────────────────────────────────────
        st.write("📥 Validating inputs…")
        ic = build_input_context(
            image_path=path_t1,
            csv_path=path_csv,
            question=user_question,
        )

        if date_t1 is not None:
            ic.acquisition_month = date_t1.month
            ic.lat = gee_params["lat"] if gee_params else getattr(ic, "lat", None)
        if date_t2 is not None:
            ic.acquisition_month_t2 = date_t2.month

        # ── Step 1b — Second image ────────────────────────────────────────────
        if path_t2:
            st.write("📅 Loading second image…")
            array_t2, meta_t2, _, _ = load_image(path_t2)
            ic.image_array_t2 = array_t2
            ic.image_meta_t2  = meta_t2

        # ── Step 2 — Image processing ─────────────────────────────────────────
        st.write("🔬 Computing spectral indices…")
        ic = process_image(ic)

        # ── Step 2b — Temporal NDVI ───────────────────────────────────────────
        if path_t2 and getattr(ic, "image_array_t2", None) is not None:
            st.write("📅 Computing temporal NDVI delta…")
            from geospatial_platform.image_processor import (
                compute_ndvi, detect_sensor, BAND_CONFIG,
                is_prescaled, get_band, normalize_to_reflectance
            )
            arr2    = ic.image_array_t2
            sensor2 = detect_sensor(arr2.shape[0])
            config2 = BAND_CONFIG[sensor2]
            pre2    = is_prescaled(arr2)

            red2 = get_band(arr2, config2, "red")
            nir2 = get_band(arr2, config2, "nir")
            if red2 is not None and nir2 is not None:
                red2 = normalize_to_reflectance(red2, pre2)
                nir2 = normalize_to_reflectance(nir2, pre2)
            ndvi_t2 = compute_ndvi(red2, nir2)

            if ndvi_t2 is not None and not np.all(np.isnan(ndvi_t2)):
                mean_t1 = float(np.nanmean(ic.ndvi)) if ic.ndvi is not None else None
                mean_t2 = float(np.nanmean(ndvi_t2))
                ic.ndvi_mean_t1 = mean_t1
                ic.ndvi_mean_t2 = mean_t2
                ic.ndvi_delta   = round(mean_t2 - mean_t1, 4) if mean_t1 is not None else None

                def _year_from_name(fname):
                    m = re.search(r'(19|20)\d{2}', fname or '')
                    return m.group(0) if m else None

                ic.temporal_label_t1 = (
                    _year_from_name(uploaded_t1.name) if uploaded_t1 else
                    str(gee_params["year1"]) if gee_params else "Image 1"
                )
                ic.temporal_label_t2 = (
                    _year_from_name(uploaded_t2.name) if uploaded_t2 else
                    str(gee_params["year2"]) if gee_params else "Image 2"
                )

                if ic.ndvi_delta is not None and abs(ic.ndvi_delta) > 0.05:
                    direction = "improvement" if ic.ndvi_delta > 0 else "decline"
                    ic.anomalies = list(ic.anomalies or [])
                    ic.anomalies.append(
                        f"vegetation {direction} detected (ΔNDVI={ic.ndvi_delta:+.3f})"
                    )

        # ── Step 3 — ViT ──────────────────────────────────────────────────────
        st.write("🧠 Extracting Vision Transformer features…")
        ic = extract_vit_features(ic)

        # ── Step 4 — Climate ──────────────────────────────────────────────────
        if path_csv:
            st.write("📊 Integrating uploaded climate CSV…")
            try:
                ic = integrate_data(ic)
                df = getattr(ic, "csv_df", None)
                if df is not None and build_climate_summary:
                    ic.climate_summary = build_climate_summary(df)
                    if populate_convenience_fields:
                        populate_convenience_fields(ic)
            except Exception as e:
                pipeline_warnings.append(f"Climate integration failed: {e}")

        elif climate_mode == "Auto-fetch (NASA POWER)":
            st.write("🌍 Auto-fetching climate data from NASA POWER…")
            try:
                from climate_fetcher import fetch_nasa_power, climate_data_quality_report
                fetch_lat = (
                    gee_params["lat"] if gee_params
                    else _extract_lat_from_meta(ic.image_meta)
                )
                fetch_lon = (
                    gee_params["lon"] if gee_params
                    else _extract_lon_from_meta(ic.image_meta)
                )
                if fetch_lat is not None and fetch_lon is not None:
                    climate_df = fetch_nasa_power(fetch_lat, fetch_lon)
                    if climate_df is not None:
                        qr = climate_data_quality_report(climate_df)
                        if not qr["suitable"]:
                            pipeline_warnings.append(
                                f"NASA POWER data quality: {qr['n_years']} years, "
                                f"{qr['missing_rain']} missing rainfall months."
                            )
                        ic.csv_df = climate_df
                        ic = integrate_data(ic)
                        if build_climate_summary:
                            ic.climate_summary = build_climate_summary(ic.csv_df)
                        if populate_convenience_fields:
                            populate_convenience_fields(ic)
                        st.write(f"  ✓ {qr['n_years']} years of climate data fetched")
                    else:
                        pipeline_warnings.append(
                            "NASA POWER returned no data — try uploading a CSV manually."
                        )
                else:
                    pipeline_warnings.append(
                        "Could not determine coordinates for NASA POWER fetch."
                    )
            except Exception as e:
                pipeline_warnings.append(f"NASA POWER auto-fetch failed: {e}")
        else:
            st.write("📊 No climate data — skipping.")

        # ── Step 5 — RAG ──────────────────────────────────────────────────────
        st.write("📚 Retrieving environmental context (RAG)…")
        ic = retrieve_context(ic)
        if not getattr(ic, "rag_context", None):
            ic.rag_context = getattr(ic, "retrieved_context", "") or ""

        # ── Step 5b — NDVI time series ────────────────────────────────────────
        ic.ndvi_timeseries  = None
        ic.ndvi_trend_stats = None
        if fetch_timeseries:
            st.write("📈 Fetching NDVI time series from GEE…")
            try:
                from time_series import (
                    fetch_ndvi_time_series, compute_trend,
                    estimate_growing_season_months
                )
                from gee_connector import init_gee
                if init_gee():
                    ts_lat = (
                        gee_params["lat"] if gee_params
                        else _extract_lat_from_meta(ic.image_meta)
                    )
                    ts_lon = (
                        gee_params["lon"] if gee_params
                        else _extract_lon_from_meta(ic.image_meta)
                    )
                    if ts_lat and ts_lon:
                        m_start, m_end = estimate_growing_season_months(
                            ts_lat, ic.aridity_index
                        )
                        ts_df = fetch_ndvi_time_series(
                            ts_lat, ts_lon,
                            start_year=2010, end_year=2024,
                            month_start=m_start, month_end=m_end,
                        )
                        if ts_df is not None:
                            ic.ndvi_timeseries  = ts_df
                            ic.ndvi_trend_stats = compute_trend(ts_df)
                            st.write(f"  ✓ {len(ts_df)} years of NDVI data fetched")
                        else:
                            pipeline_warnings.append("NDVI time series returned no data.")
                    else:
                        pipeline_warnings.append("Coordinates not available for time series.")
                else:
                    pipeline_warnings.append("GEE not initialised — time series skipped.")
            except Exception as e:
                pipeline_warnings.append(f"Time series fetch failed: {e}")

        # ── Patch derived fields ──────────────────────────────────────────────
        for attr in ("ndvi", "ndwi", "ndbi"):
            arr = getattr(ic, attr, None)
            mean_attr = f"{attr}_mean"
            map_attr  = f"{attr}_map"
            if arr is not None and getattr(ic, mean_attr, None) is None:
                setattr(ic, mean_attr, float(np.nanmean(arr)))
                setattr(ic, map_attr,  arr)

        if not getattr(ic, "region", None):
            ic.region = ic.image_meta.get("region_name", "Unknown region")
        if not getattr(ic, "ecosystem", None):
            ic.ecosystem = ic.image_meta.get("ecosystem", "Mixed landscape")

        # ── Confidence score ──────────────────────────────────────────────────
        if ic.confidence_score is None:
            score = 0.0
            if ic.ndvi is not None:                       score += 20
            if ic.ndwi is not None:                       score += 15
            if ic.ndbi is not None:                       score += 10
            if ic.aridity_index is not None:              score += 10
            if getattr(ic, "csv_df", None) is not None:  score += 20
            if ic.ndvi_delta is not None:                 score += 15
            ic.confidence_score = min(85.0, score)

        # ── Step 6 — Report ───────────────────────────────────────────────────
        st.write("✍️ Generating scientific report…")
        ic.report = generate_report(ic, ic.rag_context or "", ic.anomalies or [])

    status.update(label="✅ Analysis complete", state="complete", expanded=False)

    for w in pipeline_warnings:
        st.warning(w)

except Exception as e:
    status.update(label="❌ Pipeline error", state="error", expanded=True)
    st.markdown(f"""
    <div class="geo-card geo-card-error">
        <strong>Pipeline failed:</strong><br><br>
        <code style="font-size:0.78rem;">{type(e).__name__}: {e}</code>
    </div>
    """, unsafe_allow_html=True)
    st.exception(e)
    st.stop()

finally:
    for p in temp_files:
        try:
            os.unlink(p)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

tab_overview, tab_maps, tab_report = st.tabs([
    "  📊 Overview  ", "  🗺️ Index Maps  ", "  📄 Full Report  ",
])


# ── TAB 1 — OVERVIEW ─────────────────────────────────────────────────────────

with tab_overview:
    region_str   = getattr(ic, "region", None) or ic.image_meta.get("region_name", "Unknown region")
    eco_str      = getattr(ic, "ecosystem", None) or "—"
    t1_label     = getattr(ic, "temporal_label_t1", None)
    t2_label     = getattr(ic, "temporal_label_t2", None)
    temporal_str = f"{t1_label} → {t2_label}" if t1_label and t2_label else "Single image"

    st.markdown(f"""
    <div class="geo-card geo-card-accent">
        <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:0.5rem;">
            <div>
                <div class="label-mono">Region</div>
                <div style="font-size:1.05rem;font-weight:600;color:#1a1f2e;margin-top:0.2rem;">📍 {region_str}</div>
                <div style="font-size:0.8rem;color:#6b7280;margin-top:0.2rem;">{eco_str}</div>
            </div>
            <div style="text-align:right;">
                <div class="label-mono">Temporal Coverage</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.9rem;color:#2563eb;margin-top:0.2rem;">📅 {temporal_str}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("## Spectral Indices")
        ndvi  = getattr(ic, "ndvi_mean", None)
        ndwi  = getattr(ic, "ndwi_mean", None)
        ndbi  = getattr(ic, "ndbi_mean", None)
        ai    = getattr(ic, "aridity_index", None)
        delta = getattr(ic, "ndvi_delta", None)

        chips = ""
        if ndvi  is not None: chips += f'<div class="metric-chip">NDVI<span>{ndvi:.3f}</span></div>'
        if ndwi  is not None: chips += f'<div class="metric-chip">NDWI<span>{ndwi:.3f}</span></div>'
        if ndbi  is not None: chips += f'<div class="metric-chip">NDBI<span>{ndbi:.3f}</span></div>'
        if ai    is not None: chips += f'<div class="metric-chip">Aridity<span>{ai:.3f}</span></div>'
        if delta is not None:
            sign   = "+" if delta >= 0 else ""
            dcolor = "#16a34a" if delta >= 0 else "#dc2626"
            chips += f'<div class="metric-chip">ΔNDVI<span style="color:{dcolor}">{sign}{delta:.3f}</span></div>'
        st.markdown(f'<div class="metric-row">{chips}</div>', unsafe_allow_html=True)

        land_cover = getattr(ic, "land_cover", None)
        if land_cover:
            st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:1.05rem;font-weight:600;color:#2563eb;border-bottom:1px solid #dde3ed;padding-bottom:0.4rem;margin:1rem 0 0.6rem 0;">Land Cover</div>', unsafe_allow_html=True)
            st.markdown(render_land_cover_bar(land_cover), unsafe_allow_html=True)
            lc_cols   = st.columns(4)
            lc_colors = {"water": "#3b82f6", "vegetation": "#22c55e", "urban": "#f59e0b", "barren": "#94a3b8"}
            for i, (cls, pct) in enumerate(land_cover.items()):
                with lc_cols[i % 4]:
                    c = lc_colors.get(cls, "#94a3b8")
                    st.markdown(f"""
                    <div style="text-align:center;margin-top:0.5rem;">
                        <div style="width:12px;height:12px;background:{c};border-radius:2px;margin:0 auto 3px;"></div>
                        <div class="label-mono">{cls}</div>
                        <div style="font-family:'IBM Plex Mono',monospace;font-size:1rem;color:#1a1f2e;font-weight:600;">{pct:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

        anomalies = getattr(ic, "anomalies", None)
        if anomalies:
            st.markdown("## Detected Anomalies")
            tags = "".join(f'<span class="anomaly-tag">⚠ {a}</span>' for a in anomalies)
            st.markdown(f"<div>{tags}</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("## Confidence Score")
        score       = getattr(ic, "confidence_score", None) or 0
        label_color = "#16a34a" if score >= 75 else "#d97706" if score >= 55 else "#dc2626"
        label_text  = "High" if score >= 75 else "Moderate" if score >= 55 else "Low"
        st.markdown(f"""
        <div class="geo-card">
            <div style="display:flex;justify-content:space-between;align-items:baseline;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:2rem;color:{label_color};font-weight:600;">{score:.0f}%</div>
                <div class="label-mono">{label_text} confidence</div>
            </div>
            {render_confidence_bar(score)}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("## Image Metadata")
        meta = ic.image_meta
        meta_items = [
            ("Format",     getattr(ic, "image_format", "—")),
            ("Bands",      str(getattr(ic, "n_bands", "—"))),
            ("Dimensions", f"{meta.get('width','?')} × {meta.get('height','?')} px"),
            ("CRS",        meta.get("crs", "—")),
            ("Context",    meta.get("region_context", "—")),
        ]
        rows = "".join(f"""
        <div class="meta-row">
            <span class="label-mono">{k}</span>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;color:#374151;text-align:right;max-width:55%;">{v}</span>
        </div>""" for k, v in meta_items)
        st.markdown(f'<div class="geo-card">{rows}</div>', unsafe_allow_html=True)

        climate_summary = getattr(ic, "climate_summary", None)
        if climate_summary:
            st.markdown("## Climate Snapshot")
            cs = climate_summary
            pairs = [
                ("Rainfall (latest)", f"{cs.get('rainfall_mm_latest', 0):.1f} mm"),
                ("Rainfall trend",    cs.get("rainfall_mm_trend", "—").capitalize()),
                ("Temperature",       f"{cs.get('temperature_c_latest', 0):.1f} °C"),
                ("Humidity",          f"{cs.get('humidity_pct_latest', 0):.1f} %"),
            ]
            rows = "".join(f"""
            <div class="meta-row">
                <span class="label-mono">{k}</span>
                <span style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;color:#374151;">{v}</span>
            </div>""" for k, v in pairs)
            st.markdown(f'<div class="geo-card">{rows}</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="geo-card" style="text-align:center;color:#9ca3af;font-size:0.8rem;padding:1.5rem;">
                No climate data uploaded.
            </div>
            """, unsafe_allow_html=True)


# ── TAB 2 — INDEX MAPS ───────────────────────────────────────────────────────

with tab_maps:
    st.markdown("## Spectral Index Maps")
    ndvi_map = getattr(ic, "ndvi_map", None)
    ndwi_map = getattr(ic, "ndwi_map", None)
    ndbi_map = getattr(ic, "ndbi_map", None)

    available = [
        (ndvi_map, "NDVI — Vegetation Density", "RdYlGn"),
        (ndwi_map, "NDWI — Water Content",      "Blues_r"),
        (ndbi_map, "NDBI — Built-up Index",     "YlOrRd"),
    ]
    available = [(a, t, c) for a, t, c in available if a is not None]

    if not available:
        st.markdown('<div class="geo-card" style="text-align:center;color:#9ca3af;padding:2rem;">No index maps available.</div>', unsafe_allow_html=True)
    else:
        cols = st.columns(len(available))
        for col, (arr, title, cmap) in zip(cols, available):
            with col:
                fig = render_index_map(arr, title, cmap)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

    ndvi_delta = getattr(ic, "ndvi_delta", None)
    ndvi_t1    = getattr(ic, "ndvi_mean_t1", None)
    ndvi_t2    = getattr(ic, "ndvi_mean_t2", None)
    if ndvi_delta is not None and ndvi_t1 is not None and ndvi_t2 is not None:
        st.markdown("## Temporal NDVI Comparison")
        card_cls    = "geo-card-good" if ndvi_delta >= 0 else "geo-card-error"
        arrow       = f"↑ +{ndvi_delta:.3f}" if ndvi_delta >= 0 else f"↓ {ndvi_delta:.3f}"
        arrow_color = "#16a34a" if ndvi_delta >= 0 else "#dc2626"
        st.markdown(f"""
        <div class="geo-card {card_cls}">
            <div style="display:flex;gap:2.5rem;align-items:center;flex-wrap:wrap;">
                <div><div class="label-mono">{t1_label or 'Image 1'}</div>
                     <div style="font-family:'IBM Plex Mono',monospace;font-size:1.4rem;color:#1a1f2e;">{ndvi_t1:.3f}</div></div>
                <div style="font-size:1.4rem;color:#9ca3af;">→</div>
                <div><div class="label-mono">{t2_label or 'Image 2'}</div>
                     <div style="font-family:'IBM Plex Mono',monospace;font-size:1.4rem;color:#1a1f2e;">{ndvi_t2:.3f}</div></div>
                <div style="margin-left:1rem;">
                    <div class="label-mono">Change (ΔNDVI)</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:1.6rem;color:{arrow_color};font-weight:700;">{arrow}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    delta_map = getattr(ic, "ndvi_trend_map", None)
    if delta_map is not None:
        st.markdown("## ΔNDVI Spatial Change Map")
        st.markdown('<div style="font-size:0.8rem;color:#6b7280;margin-bottom:0.5rem;">Green = vegetation gain · Red = vegetation loss</div>', unsafe_allow_html=True)
        fig_delta = render_index_map(delta_map, "ΔNDVI Change Map (pixel-wise)", "RdYlGn")
        st.pyplot(fig_delta, use_container_width=True)
        plt.close(fig_delta)

    ts_df    = getattr(ic, "ndvi_timeseries", None)
    ts_trend = getattr(ic, "ndvi_trend_stats", None)
    if ts_df is not None and len(ts_df) > 1:
        st.markdown("## NDVI Time Series (Annual)")
        try:
            from time_series import render_time_series_chart
            fig_ts = render_time_series_chart(
                ts_df, ts_trend or {},
                ecosystem=getattr(ic, "ecosystem", "") or ""
            )
            st.pyplot(fig_ts, use_container_width=True)
            plt.close(fig_ts)
            if ts_trend:
                trend_color = "#16a34a" if (ts_trend.get("slope") or 0) > 0 else "#dc2626"
                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-chip">Trend<span>{ts_trend.get('trend','—')}</span></div>
                    <div class="metric-chip">Rate<span style="color:{trend_color}">{ts_trend.get('annual_rate',0):+.4f}/yr</span></div>
                    <div class="metric-chip">Total Δ<span style="color:{trend_color}">{ts_trend.get('total_change',0):+.3f}</span></div>
                    <div class="metric-chip">R²<span>{ts_trend.get('r2',0):.2f}</span></div>
                </div>""", unsafe_allow_html=True)
        except Exception as e:
            st.caption(f"Time series chart error: {e}")


# ── TAB 3 — FULL REPORT ──────────────────────────────────────────────────────

with tab_report:
    report = getattr(ic, "report", None)
    if not report:
        st.markdown('<div class="geo-card" style="text-align:center;color:#9ca3af;padding:2rem;">Report not generated.</div>', unsafe_allow_html=True)
    else:
        first_section = report.find("## 1.")
        if first_section == -1:
            first_section = report.find("## ")
        report_body = report[first_section:] if first_section != -1 else report

        footer_match = re.search(r'\n={10,}\s*\nConfidence:', report_body)
        if footer_match:
            report_body = report_body[:footer_match.start()].strip()
        else:
            report_body = re.sub(r'\n={10,}.*$', '', report_body, flags=re.DOTALL).strip()

        sections = parse_report_sections(report_body)
        if sections:
            for title, content in sections.items():
                render_report_section(title, content, get_icon(title))
        else:
            st.markdown(f'<div class="report-section" style="white-space:pre-wrap;">{report}</div>', unsafe_allow_html=True)

        st.markdown("<hr class='geo-divider'>", unsafe_allow_html=True)
        col_dl, _ = st.columns([1, 3])
        with col_dl:
            st.download_button(
                label="⬇ Download Report (.txt)",
                data=report,
                file_name=f"geointel_{region_str.replace(' ', '_').replace(',', '')}.txt",
                mime="text/plain",
            )
```
